import os
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from agents.base_agent import BaseAgent


# ============================================================
# ニューラルネットワーク（PyTorch）
# ============================================================

class QNetwork(nn.Module):
    """
    CNN版: 入力 shape=(B, 3, 5, 5) → Conv2d×2 → flatten → FC → Q値 (B, 5)

    Conv2d(3→32, 3×3, pad=1) → ReLU → Conv2d(32→64, 3×3, pad=1) → ReLU
    → flatten(64*5*5=1600) → Linear(1600→256) → ReLU → Linear(256→5)
    """

    def __init__(self, rows=5, cols=5, output_size=5):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * rows * cols, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )
        # He初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


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
    Connect Three 用 DQN エージェント（PyTorch版）。
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
        self.action_size = 5
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval
        self.total_steps = 0

        self.device = torch.device('cpu')

        self.qnet        = QNetwork(rows=5, cols=5, output_size=5).to(self.device)
        self.qnet_target = QNetwork(rows=5, cols=5, output_size=5).to(self.device)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.qnet_target.eval()

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def get_action(self, state, valid_actions):
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.qnet(state_t)[0].numpy()

        masked_q = np.full(self.action_size, -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]
        return int(np.argmax(masked_q))

    def on_step_end(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1

        if self.total_steps < self.warmup_steps:
            return None
        if len(self.replay_buffer) < self.batch_size:
            return None

        loss = self._train_step()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        if self.total_steps % self.target_update_interval == 0:
            self.qnet_target.load_state_dict(self.qnet.state_dict())

        return loss

    def _train_step(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t      = torch.from_numpy(states).to(self.device)
        actions_t     = torch.from_numpy(actions).long().to(self.device)
        rewards_t     = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t       = torch.from_numpy(dones).to(self.device)

        current_q = self.qnet(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.qnet_target(next_states_t).max(dim=1).values
            target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        torch.save(self.qnet.state_dict(), path + '.pt')

    def load(self, path):
        self.qnet.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def save_checkpoint(self, path):
        self.save(path)
        ckpt = {
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        buf_data = self.replay_buffer.get_data_for_save()
        ckpt['buffer'] = buf_data
        torch.save(ckpt, path + '_checkpoint.pt')

    def load_checkpoint(self, path, load_buffer=True):
        self.load(path + '.pt')
        ckpt_path = path + '_checkpoint.pt'
        if not os.path.exists(ckpt_path):
            return False
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if 'epsilon' in ckpt:
            self.epsilon = float(ckpt['epsilon'])
        if 'total_steps' in ckpt:
            self.total_steps = int(ckpt['total_steps'])
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if load_buffer and 'buffer' in ckpt and ckpt['buffer']:
            self.replay_buffer.load_from_data(ckpt['buffer'])
        return True
