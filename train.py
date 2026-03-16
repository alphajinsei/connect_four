"""
train.py — DQN エージェントの学習スクリプト（再設計版）

設計方針:
  - DQN は常に PLAYER1（先手）として学習
  - 対戦相手はルールベースAI（勝ち手・負け手阻止・中央優先）
  - eval毎に vs RuleBased の勝率を表示（絶対的な強さの指標）
  - 将来的に --selfplay でプール方式に移行できる拡張口を残す

使い方:
    # 基本: ルールベースAI相手にゼロから学習
    python train.py --episodes 20000

    # 学習済み重みから続き
    python train.py --load-path weights/dqn_connect4 --episodes 20000

    # エピソード数・評価間隔を指定
    python train.py --episodes 20000 --eval-interval 500
"""
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from env.connect4_env import Connect4Env
from agents.dqn_agent import DQNAgent
from agents.rule_based_agent import RuleBasedAgent
from game_runner import GameRunner

WEIGHTS_PATH  = "weights/dqn_connect4"
SNAPSHOTS_DIR = "weights/snapshots"
LOG_PATH      = "weights/train_log.txt"


class Tee:
    """stdout とファイルに同時に書き出す"""
    def __init__(self, path):
        self._file   = open(path, "w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def make_agent(**kwargs):
    defaults = dict(
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99990,       # ~30000ステップ(≒3750ep)でε=0.05に到達
        buffer_capacity=50000,
        batch_size=128,
        warmup_steps=2000,
        target_update_interval=200,
    )
    defaults.update(kwargs)
    return DQNAgent(**defaults)


def eval_vs_rulebased(agent, env, n=200):
    """ルールベースAI と n 戦して勝率を返す（greedy、学習なし）"""
    runner     = GameRunner(env, agent, RuleBasedAgent(), renderer=None)
    saved_eps  = agent.epsilon
    agent.epsilon = 0.0
    wins = sum(
        1 for _ in range(n)
        if runner.run_episode()["winner"] == Connect4Env.PLAYER1
    )
    agent.epsilon = saved_eps
    return wins / n * 100


def print_header():
    print(f"{'Episode':>8} | {'勝率(直近500)':>14} | {'平均報酬':>10} | {'ε':>7} | {'vs RuleBased(200戦)':>20}")
    print("-" * 72)


def train(num_episodes=20000, eval_interval=500, load_path=None):
    os.makedirs("weights",      exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR,  exist_ok=True)

    tee        = Tee(LOG_PATH)
    sys.stdout = tee

    env   = Connect4Env()

    if load_path:
        agent = make_agent(epsilon_start=0.10)
        agent.load(load_path + ".npz")
        print(f"重みをロード: {load_path}.npz  (ε={agent.epsilon:.4f})")
    else:
        agent = make_agent()
        print("新規学習開始")

    opponent = RuleBasedAgent()
    print("対戦相手: ルールベースAI（先手固定で学習）")
    print()

    runner = GameRunner(env, agent, opponent, renderer=None)

    win_history    = []
    reward_history = []
    best_vs_rb     = 0.0

    print_header()

    for episode in range(1, num_episodes + 1):
        stats = runner.run_episode()
        win_history.append(1 if stats["winner"] == Connect4Env.PLAYER1 else 0)
        reward_history.append(stats["reward_p1"])

        if episode % eval_interval == 0:
            win_rate   = np.mean(win_history[-500:])  * 100
            avg_reward = np.mean(reward_history[-500:])
            vs_rb      = eval_vs_rulebased(agent, env)

            print(f"{episode:>8} | {win_rate:>13.1f}% | {avg_reward:>10.3f} | {agent.epsilon:>7.5f} | {vs_rb:>19.1f}%")

            # 自己ベスト更新時にスナップショット保存
            if vs_rb > best_vs_rb:
                best_vs_rb = vs_rb
                snap_path  = os.path.join(SNAPSHOTS_DIR, f"best_ep{episode}_rb{vs_rb:.0f}pct")
                agent.save(snap_path)
                print(f"  [Best] vs RuleBased {vs_rb:.1f}% → スナップショット保存: {snap_path}.npz")

    print("\n学習完了")
    agent.save(WEIGHTS_PATH)
    print(f"重みを保存しました: {WEIGHTS_PATH}.npz")

    print("\n=== 勝率推移（直近500エピソード平均）===")
    window = 500
    for i in range(window, len(win_history) + 1, window):
        wr  = np.mean(win_history[i-window:i]) * 100
        bar = "#" * int(wr / 2)
        print(f"Ep {i:>6}: {bar:<50} {wr:.1f}%")

    sys.stdout = tee._stdout
    tee.close()
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",      type=int, default=20000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--load-path",     type=str, default=None,
                        help="学習済み重みから再開 例: weights/dqn_connect4")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        load_path=args.load_path,
    )
