import os
import numpy as np
import random
from collections import deque
from agents.base_agent import BaseAgent


# ============================================================
# ニューラルネットワーク（NumPy手書き）
# ============================================================

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout = dout.copy()
        dout[self.mask] = 0
        return dout


class Linear:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T


class AdamOptimizer:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def update(self, params, grads):
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class QNetwork:
    """
    入力: shape=(3, 6, 7) → flatten → 126次元
    出力: 各列のQ値 shape=(7,)
    構造: Linear(126→256) -> ReLU -> Linear(256→256) -> ReLU -> Linear(256→7)
    """

    def __init__(self, input_size=126, hidden_size=256, output_size=7, lr=1e-3):
        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size),
            'b1': np.zeros(hidden_size),
            'W2': np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size),
            'b2': np.zeros(hidden_size),
            'W3': np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size),
            'b3': np.zeros(output_size),
        }
        self._build_layers()
        self.optimizer = AdamOptimizer(lr=lr)

    def _build_layers(self):
        self.layers = [
            Linear(self.params['W1'], self.params['b1']),
            Relu(),
            Linear(self.params['W2'], self.params['b2']),
            Relu(),
            Linear(self.params['W3'], self.params['b3']),
        ]

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, target_q):
        q = self.predict(x)
        diff = q - target_q
        loss_val = np.mean(diff ** 2)

        dout = 2 * diff / diff.size
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        grads = {
            'W1': self.layers[0].dW, 'b1': self.layers[0].db,
            'W2': self.layers[2].dW, 'b2': self.layers[2].db,
            'W3': self.layers[4].dW, 'b3': self.layers[4].db,
        }
        self.optimizer.update(self.params, grads)

        # Linear層の参照を更新後の値に同期
        self.layers[0].W = self.params['W1'];  self.layers[0].b = self.params['b1']
        self.layers[2].W = self.params['W2'];  self.layers[2].b = self.params['b2']
        self.layers[4].W = self.params['W3'];  self.layers[4].b = self.params['b3']

        return loss_val

    def copy_weights_from(self, other):
        for key in self.params:
            self.params[key] = other.params[key].copy()
        self.layers[0].W = self.params['W1'];  self.layers[0].b = self.params['b1']
        self.layers[2].W = self.params['W2'];  self.layers[2].b = self.params['b2']
        self.layers[4].W = self.params['W3'];  self.layers[4].b = self.params['b3']

    def save(self, path):
        np.savez(path, **self.params)

    def load(self, path):
        data = np.load(path)
        for key in self.params:
            self.params[key] = data[key]
        self.layers[0].W = self.params['W1'];  self.layers[0].b = self.params['b1']
        self.layers[2].W = self.params['W2'];  self.layers[2].b = self.params['b2']
        self.layers[4].W = self.params['W3'];  self.layers[4].b = self.params['b3']


# ============================================================
# 経験再生バッファ
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int32),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def get_data_for_save(self):
        """バッファの内容を numpy 配列として返す（保存用）"""
        if len(self.buffer) == 0:
            return {}
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        return {
            'buf_states': np.array(states, dtype=np.float32),
            'buf_actions': np.array(actions, dtype=np.int32),
            'buf_rewards': np.array(rewards, dtype=np.float32),
            'buf_next_states': np.array(next_states, dtype=np.float32),
            'buf_dones': np.array(dones, dtype=np.float32),
        }

    def load_from_data(self, data):
        """numpy 配列からバッファを復元する"""
        self.buffer.clear()
        n = len(data['buf_states'])
        for i in range(n):
            self.buffer.append((
                data['buf_states'][i],
                int(data['buf_actions'][i]),
                float(data['buf_rewards'][i]),
                data['buf_next_states'][i],
                float(data['buf_dones'][i]),
            ))

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DQNエージェント
# ============================================================

class DQNAgent(BaseAgent):
    """
    Connect Four 用 DQN エージェント。

    BaseAgent インターフェース:
        get_action(state, valid_actions) -> int
        on_step_end(state, action, reward, next_state, done)
    """

    def __init__(
        self,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9999,
        buffer_capacity=20000,
        batch_size=64,
        warmup_steps=1000,
        target_update_interval=500,
    ):
        self.action_size = 7
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval
        self.total_steps = 0

        self.qnet        = QNetwork(input_size=126, hidden_size=256, output_size=7, lr=lr)
        self.qnet_target = QNetwork(input_size=126, hidden_size=256, output_size=7, lr=lr)
        self.qnet_target.copy_weights_from(self.qnet)

        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def get_action(self, state, valid_actions):
        """ε-greedy + 無効手マスキングで列を選択"""
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)

        state_flat = state.flatten()[np.newaxis, :]      # (1, 126)
        q_values = self.qnet.predict(state_flat)[0]      # (7,)

        masked_q = np.full(self.action_size, -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]
        return int(np.argmax(masked_q))

    def on_step_end(self, state, action, reward, next_state, done):
        """経験をバッファに追加し、バッチ学習を行う"""
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1

        # ウォームアップ中はバッファ蓄積のみ（学習しない）
        if self.total_steps < self.warmup_steps:
            return None
        if len(self.replay_buffer) < self.batch_size:
            return None

        loss = self._train_step()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        if self.total_steps % self.target_update_interval == 0:
            self.qnet_target.copy_weights_from(self.qnet)

        return loss

    def _train_step(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_flat      = states.reshape(self.batch_size, -1)       # (B, 126)
        next_states_flat = next_states.reshape(self.batch_size, -1)  # (B, 126)

        # ターゲットQ値: r + γ * max Q_target(s', a')
        # s' は「相手が1手打った後の自分のターン」の状態。
        # GameRunner が遅延コールバックで 2手後の状態を next_state として渡すため、
        # 通常の DQN 更新式がそのまま使える。
        next_q = self.qnet_target.predict(next_states_flat)          # (B, 7)
        max_next_q = np.max(next_q, axis=1)                          # (B,)
        target_q_values = rewards + self.gamma * max_next_q * (1 - dones)

        current_q = self.qnet.predict(states_flat)                   # (B, 7)
        target_q = current_q.copy()
        for i in range(self.batch_size):
            target_q[i, actions[i]] = target_q_values[i]

        return self.qnet.loss(states_flat, target_q)

    def save(self, path):
        """重みを保存する（path に .npz 拡張子不要）"""
        self.qnet.save(path)

    def load(self, path):
        """重みを読み込む"""
        self.qnet.load(path)
        self.qnet_target.copy_weights_from(self.qnet)

    def save_checkpoint(self, path):
        """重み + バッファ + 学習状態を保存する（学習再開用）

        重みは path.npz に、チェックポイントは path_checkpoint.npz に保存。
        重みファイルは WebUI やスナップショットプールで使うため軽量のまま。
        チェックポイントはバッファ・ε・Adam状態を含み、学習再開時のみ使用。
        """
        self.qnet.save(path)

        ckpt = {
            'epsilon': np.array(self.epsilon),
            'total_steps': np.array(self.total_steps),
            'adam_t': np.array(self.qnet.optimizer.t),
        }
        for key in self.qnet.optimizer.m:
            ckpt[f'adam_m_{key}'] = self.qnet.optimizer.m[key]
            ckpt[f'adam_v_{key}'] = self.qnet.optimizer.v[key]

        buf_data = self.replay_buffer.get_data_for_save()
        ckpt.update(buf_data)

        np.savez_compressed(path + '_checkpoint', **ckpt)

    def load_checkpoint(self, path, load_buffer=True):
        """重み + 学習状態（+ オプションでバッファ）を復元する

        Args:
            path: 拡張子なしのパス（path.npz と path_checkpoint.npz を読む）
            load_buffer: True ならバッファも復元、False なら重み+状態のみ
        """
        self.qnet.load(path + '.npz')
        self.qnet_target.copy_weights_from(self.qnet)

        ckpt_path = path + '_checkpoint.npz'
        if not os.path.exists(ckpt_path):
            return False

        data = np.load(ckpt_path)

        if 'epsilon' in data:
            self.epsilon = float(data['epsilon'])
        if 'total_steps' in data:
            self.total_steps = int(data['total_steps'])

        if 'adam_t' in data:
            self.qnet.optimizer.t = int(data['adam_t'])
            for key in self.qnet.params:
                m_key = f'adam_m_{key}'
                v_key = f'adam_v_{key}'
                if m_key in data:
                    self.qnet.optimizer.m[key] = data[m_key]
                    self.qnet.optimizer.v[key] = data[v_key]

        if load_buffer and 'buf_states' in data:
            self.replay_buffer.load_from_data(data)

        return True
