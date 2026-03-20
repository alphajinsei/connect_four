"""
train.py — AlphaZero 学習スクリプト

=== AlphaZeroの学習サイクル（DQNとの対比） ===

【DQN】
  1. 相手（ルールベースAI等）と1手ずつ対戦
  2. 各手で (state, action, reward, next_state) を保存
  3. ミニバッチで Q(s,a) = r + γ*max Q(s',a') を学習
  → 問題: 1手先のQ値しか見ない。相手の戦略を知らない。

【AlphaZero】
  1. Self-play: MCTSで自分自身と対戦し、1ゲーム分のデータを生成
     → 各手で「MCTSの探索結果 π（行動確率）」を記録
  2. ゲーム終了後、勝敗 z を全ての手に付与
  3. NNを (state → π, z) で学習
     → Policy: 「MCTSが選んだ手の分布」を模倣
     → Value: 「この局面の勝率」を予測
  4. 学習後のNNでまたself-play → 繰り返し

【DQNが失敗した理由とAlphaZeroが解決する仕組み】
  - DQN: 「盤面だけ」を見て手を決める（マルコフ性の仮定）
    → 相手がランダムか強いかを区別できない
  - AlphaZero: MCTSが「相手も最善手を打つ前提」で数十手先を探索
    → 相手の戦略を仮定する必要がない（ミニマックス的に最善手を探す）
    → 1つの方策で、どんな相手にも対応できる

CPU環境向けの軽量設定:
  - num_simulations=50（本家は800）
  - self-play 10ゲーム/イテレーション（本家は25000）
  - 学習イテレーション 100回で数時間を目標

使い方:
    # 新規学習
    .venv/Scripts/python alphazero/train.py --iterations 100

    # 途中から再開（チェックポイントから自動復元）
    .venv/Scripts/python alphazero/train.py --resume

    # パラメータを変えて再開（イテレーション番号はチェックポイントから復元）
    .venv/Scripts/python alphazero/train.py --resume --iterations 200 --games-per-iter 50
"""
import sys
import os
import argparse
import time
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))

from env.connect4_env import Connect4Env
from network import AlphaZeroNet
from mcts import MCTS
from self_play import self_play_batch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(_SCRIPT_DIR, "weights")
SNAPSHOTS_DIR = os.path.join(WEIGHTS_DIR, "snapshots")
LOG_PATH = os.path.join(WEIGHTS_DIR, "train_log.txt")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "checkpoint.pt")


class Tee:
    """stdout とファイルに同時に書き出す"""
    def __init__(self, path, mode="w"):
        self._path = path
        self._stdout = sys.stdout
        with open(path, mode, encoding="utf-8") as f:
            pass

    def write(self, data):
        try:
            self._stdout.write(data)
        except UnicodeEncodeError:
            self._stdout.write(data.encode(self._stdout.encoding or 'ascii', errors='replace').decode(self._stdout.encoding or 'ascii'))
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(data)
            f.flush()

    def flush(self):
        self._stdout.flush()

    def close(self):
        pass


class ReplayBuffer:
    """
    学習データバッファ（直近Nゲーム分を保持）

    【DQNとの違い】
    DQN: (s, a, r, s') を1手ずつ保存 → TD学習
    AlphaZero: (s, π, z) を保存 → 教師あり学習に近い
      - π: MCTSが出した行動確率（「こう打つべき」という教師信号）
      - z: 実際の勝敗結果（「この局面は有利/不利」という教師信号）
    """

    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.states = []
        self.policies = []
        self.values = []

    def add(self, states, policies, values):
        self.states.extend(states)
        self.policies.extend(policies)
        self.values.extend(values)
        # 古いデータを削除
        if len(self.states) > self.max_size:
            excess = len(self.states) - self.max_size
            self.states = self.states[excess:]
            self.policies = self.policies[excess:]
            self.values = self.values[excess:]

    def sample(self, batch_size):
        indices = np.random.choice(len(self.states), size=min(batch_size, len(self.states)), replace=False)
        return (
            np.array([self.states[i] for i in indices], dtype=np.float32),
            np.array([self.policies[i] for i in indices], dtype=np.float32),
            np.array([self.values[i] for i in indices], dtype=np.float32),
        )

    def get_state(self):
        """バッファの内容をシリアライズ可能な形式で返す"""
        return {
            'states': self.states,
            'policies': self.policies,
            'values': self.values,
        }

    def load_state(self, state):
        """シリアライズされた状態からバッファを復元"""
        self.states = state['states']
        self.policies = state['policies']
        self.values = state['values']

    def __len__(self):
        return len(self.states)


def save_checkpoint(network, optimizer, buffer, iteration, best_vs_rb, session_ts):
    """学習状態を丸ごと保存（PC停止からの復帰用）"""
    checkpoint = {
        'iteration': iteration,
        'network_state': network.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'buffer_state': buffer.get_state(),
        'best_vs_rb': best_vs_rb,
        'session_ts': session_ts,
    }
    # 一時ファイルに書いてからリネーム（書き込み中の破損防止）
    tmp_path = CHECKPOINT_PATH + ".tmp"
    torch.save(checkpoint, tmp_path)
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    os.rename(tmp_path, CHECKPOINT_PATH)


def load_checkpoint(network, optimizer, buffer):
    """チェックポイントから学習状態を復元"""
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    network.load_state_dict(checkpoint['network_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    buffer.load_state(checkpoint['buffer_state'])
    return checkpoint


def train_network(network, optimizer, buffer, batch_size=128, train_steps=10):
    """
    NNの学習（1イテレーション分）

    【損失関数】（DQNとの対比）

    DQN:
      loss = (Q(s,a) - (r + γ*max Q(s',a')))²
      → 1つのスカラー値（Q値）を学習

    AlphaZero:
      loss = value_loss + policy_loss
      value_loss  = MSE(v, z)        ← 勝率予測と実際の勝敗
      policy_loss = -π・log(p)       ← MCTSの行動確率とNNの出力のクロスエントロピー
      → 2つの出力を同時に学習

    なぜこれで強くなるのか:
      1. Self-playでMCTSが良い手を探索 → πが教師信号になる
      2. NNがπを模倣できるようになる → MCTSの初手（prior）が良くなる
      3. MCTSの初手が良い → 少ないシミュレーションでも良い手を見つける
      4. さらに良いπが生成される → 2に戻る（正のフィードバックループ）
    """
    network.train()
    total_loss = 0.0
    total_v_loss = 0.0
    total_p_loss = 0.0

    for _ in range(train_steps):
        states, target_policies, target_values = buffer.sample(batch_size)

        states_t = torch.from_numpy(states).float()
        target_p = torch.from_numpy(target_policies).float()
        target_v = torch.from_numpy(target_values).float()

        # Forward
        policy_logits, value_pred = network(states_t)

        # Value loss: MSE
        value_loss = nn.functional.mse_loss(value_pred, target_v)

        # Policy loss: Cross-entropy（MCTSの行動確率 vs NNのsoftmax出力）
        log_probs = nn.functional.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.mean(torch.sum(target_p * log_probs, dim=1))

        loss = value_loss + policy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_v_loss += value_loss.item()
        total_p_loss += policy_loss.item()

    return total_loss / train_steps, total_v_loss / train_steps, total_p_loss / train_steps


def eval_vs_rulebased(network, num_games=50, num_simulations=50):
    """ルールベースAIとの対戦評価"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '4moku'))
    from agents.rule_based_agent import RuleBasedAgent

    mcts = MCTS(network, num_simulations=num_simulations, c_puct=1.5)
    rb = RuleBasedAgent()

    wins = 0
    draws = 0

    for game_i in range(num_games):
        env = Connect4Env()
        env.reset()

        while not env.done:
            if env.current_player == Connect4Env.PLAYER1:
                # AlphaZero（先手）
                action_probs = mcts.search(env, add_noise=False)
                action = int(np.argmax(action_probs))
            else:
                # RuleBased（後手）
                state_rb = env.get_canonical_state()
                valid = env.get_valid_actions()
                action = rb.get_action(state_rb, valid)

            env.step(action)

        if env.winner == Connect4Env.PLAYER1:
            wins += 1
        elif env.winner == 0:
            draws += 1

    return wins / num_games * 100, draws / num_games * 100


def eval_vs_random(network, num_games=50, num_simulations=50):
    """ランダムAIとの対戦評価"""
    mcts = MCTS(network, num_simulations=num_simulations, c_puct=1.5)
    wins = 0

    for _ in range(num_games):
        env = Connect4Env()
        env.reset()

        while not env.done:
            if env.current_player == Connect4Env.PLAYER1:
                action_probs = mcts.search(env, add_noise=False)
                action = int(np.argmax(action_probs))
            else:
                valid = env.get_valid_actions()
                action = np.random.choice(valid)

            env.step(action)

        if env.winner == Connect4Env.PLAYER1:
            wins += 1

    return wins / num_games * 100


def train(iterations=100, games_per_iter=10, num_simulations=50,
          eval_interval=5, batch_size=128, train_steps=10,
          lr=1e-3, load_path=None, resume=False):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ネットワーク・オプティマイザ・バッファを初期化
    network = AlphaZeroNet(num_res_blocks=4, channels=64)
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
    buffer = ReplayBuffer(max_size=50000)
    best_vs_rb = 0.0
    start_iteration = 1

    # 再開モード: チェックポイントから全状態を復元
    if resume:
        checkpoint = load_checkpoint(network, optimizer, buffer)
        if checkpoint:
            start_iteration = checkpoint['iteration'] + 1
            best_vs_rb = checkpoint['best_vs_rb']
            session_ts = checkpoint['session_ts']

            # ログは追記モード
            tee = Tee(LOG_PATH, mode="a")
            sys.stdout = tee

            print()
            print(f"=== 再開: イテレーション {start_iteration} から (バッファ: {len(buffer)}手) ===")
            print(f"パラメータ変更: games-per-iter={games_per_iter}, train-steps={train_steps}")
            print()
        else:
            print("チェックポイントが見つかりません。新規学習を開始します。")
            resume = False

    if not resume:
        # 新規学習 or --load からの開始
        if load_path:
            network.load_state_dict(torch.load(load_path, map_location='cpu', weights_only=True))

        tee = Tee(LOG_PATH, mode="w")
        sys.stdout = tee

        print("=" * 70)
        print("AlphaZero Connect Four -- 学習開始")
        print("=" * 70)
        print(f"イテレーション数: {iterations}")
        print(f"self-playゲーム数/イテレーション: {games_per_iter}")
        print(f"MCTSシミュレーション数: {num_simulations}")
        print(f"学習ステップ/イテレーション: {train_steps}")
        print(f"バッチサイズ: {batch_size}")
        print(f"学習率: {lr}")
        print()
        if load_path:
            print(f"重みをロード: {load_path}")
        else:
            print("新規学習（ランダム初期化）")

        param_count = sum(p.numel() for p in network.parameters())
        print(f"パラメータ数: {param_count:,}")
        print()

    print(f"{'Iter':>5} | {'Self-Play':>10} | {'Buffer':>7} | {'Loss':>8} | {'V-Loss':>8} | {'P-Loss':>8} | {'vs Random':>10} | {'vs RuleBased':>13} | {'Time':>8}")
    print("-" * 105)

    for iteration in range(start_iteration, iterations + 1):
        iter_start = time.time()

        # === 1. Self-play: MCTSで自分自身と対戦、学習データを生成 ===
        network.eval()
        states, policies, values, sp_stats = self_play_batch(
            network,
            num_games=games_per_iter,
            num_simulations=num_simulations,
            c_puct=1.5,
            temperature_threshold=15,
        )
        buffer.add(states.tolist(), policies.tolist(), values.tolist())

        sp_summary = f"P1:{sp_stats['p1_wins']} P2:{sp_stats['p2_wins']} D:{sp_stats['draws']}"

        # === 2. 学習: バッファからサンプリングしてNN更新 ===
        loss, v_loss, p_loss = 0.0, 0.0, 0.0
        if len(buffer) >= batch_size:
            loss, v_loss, p_loss = train_network(
                network, optimizer, buffer,
                batch_size=batch_size, train_steps=train_steps,
            )

        iter_time = time.time() - iter_start

        # === 3. 評価 ===
        if iteration % eval_interval == 0 or iteration == start_iteration:
            network.eval()
            vs_random = eval_vs_random(network, num_games=30, num_simulations=num_simulations)
            vs_rb, vs_rb_draw = eval_vs_rulebased(network, num_games=30, num_simulations=num_simulations)

            print(f"{iteration:>5} | {sp_summary:>10} | {len(buffer):>7} | {loss:>8.4f} | {v_loss:>8.4f} | {p_loss:>8.4f} | {vs_random:>9.1f}% | {vs_rb:>8.1f}%({vs_rb_draw:.0f}%D) | {iter_time:>7.1f}s")

            if vs_rb > best_vs_rb:
                best_vs_rb = vs_rb
                snap_path = os.path.join(SNAPSHOTS_DIR, f"best_iter{iteration}_rb{vs_rb:.0f}pct_{session_ts}.pt")
                torch.save(network.state_dict(), snap_path)
                print(f"  [Best] vs RuleBased {vs_rb:.1f}% -> {snap_path}")
        else:
            print(f"{iteration:>5} | {sp_summary:>10} | {len(buffer):>7} | {loss:>8.4f} | {v_loss:>8.4f} | {p_loss:>8.4f} | {'---':>10} | {'---':>13} | {iter_time:>7.1f}s")

        # === 4. チェックポイント保存（5イテレーションごと + 最新重み） ===
        if iteration % 5 == 0:
            save_checkpoint(network, optimizer, buffer, iteration, best_vs_rb, session_ts)
            save_path = os.path.join(WEIGHTS_DIR, "alphazero_latest.pt")
            torch.save(network.state_dict(), save_path)

    # 最終保存
    final_path = os.path.join(WEIGHTS_DIR, "alphazero_latest.pt")
    torch.save(network.state_dict(), final_path)
    save_checkpoint(network, optimizer, buffer, iterations, best_vs_rb, session_ts)
    print(f"\n学習完了。重みを保存: {final_path}")
    print(f"vs RuleBased ベスト: {best_vs_rb:.1f}%")

    sys.stdout = tee._stdout
    tee.close()
    return network


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero Connect Four Training")
    parser.add_argument("--iterations", type=int, default=100,
                        help="学習イテレーション数 (default: 100)")
    parser.add_argument("--games-per-iter", type=int, default=10,
                        help="1イテレーションあたりのself-playゲーム数 (default: 10)")
    parser.add_argument("--num-simulations", type=int, default=50,
                        help="MCTSのシミュレーション回数/手 (default: 50)")
    parser.add_argument("--eval-interval", type=int, default=5,
                        help="評価間隔 (default: 5)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="学習バッチサイズ (default: 128)")
    parser.add_argument("--train-steps", type=int, default=10,
                        help="1イテレーションの学習ステップ数 (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学習率 (default: 1e-3)")
    parser.add_argument("--load", type=str, default=None,
                        help="学習済み重みのパス（新規学習時のみ）")
    parser.add_argument("--resume", action="store_true",
                        help="チェックポイントから学習を再開")
    args = parser.parse_args()

    train(
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        num_simulations=args.num_simulations,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        lr=args.lr,
        load_path=args.load,
        resume=args.resume,
    )
