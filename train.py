"""
train.py — DQN エージェントの学習スクリプト

使い方:
    # ステージ1: ランダム相手で学習
    python train.py

    # ステージ2: カリキュラム学習（自動で相手を更新）
    python train.py --curriculum

    # 特定の重みを対戦相手にして学習（手動カリキュラム）
    python train.py --opponent-path weights/snapshot_ep5000

    # エピソード数・評価間隔を指定
    python train.py --episodes 20000 --eval-interval 500
"""
import sys
import os
import argparse
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


class Tee:
    """stdout とファイルに同時に書き出す"""
    def __init__(self, path):
        self._file = open(path, "w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()

from env.connect4_env import Connect4Env
from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from game_runner import GameRunner

WEIGHTS_PATH    = "weights/dqn_connect4"
SNAPSHOTS_DIR   = "weights/snapshots"
LOG_PATH        = "weights/train_log.txt"


def make_agent(**kwargs):
    return DQNAgent(
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99990,   # ~30000ステップ(≒3750ep)でε=0.05に到達
        buffer_capacity=50000,
        batch_size=128,
        warmup_steps=2000,
        target_update_interval=200,
        **kwargs,
    )


def print_header():
    print(f"{'Episode':>8} | {'相手':>18} | {'勝率(直近500)':>14} | {'平均報酬':>10} | {'ε':>7}")
    print("-" * 70)


def train(num_episodes=10000, eval_interval=500, opponent_path=None, curriculum=False):
    os.makedirs("weights", exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    tee = Tee(LOG_PATH)
    sys.stdout = tee

    env   = Connect4Env()
    agent = make_agent()

    # --- 対戦相手の初期化 ---
    if opponent_path:
        opponent = DQNAgent()
        opponent.load(opponent_path + ".npz")
        opponent.epsilon = 0.0  # 推論のみ（探索なし）
        opponent_label = os.path.basename(opponent_path)
        print(f"対戦相手: DQN ({opponent_label})")
    else:
        opponent = RandomAgent()
        opponent_label = "random"
        print(f"対戦相手: ランダムAI")

    if curriculum:
        print("カリキュラム学習モード: 勝率しきい値を超えると相手を自動更新")

    runner = GameRunner(env, agent, opponent, renderer=None)

    win_history    = []
    reward_history = []
    snapshot_count = 0

    # カリキュラム学習のフェーズ設定
    # (しきい値, 次フェーズの説明文)
    CURRICULUM_PHASES = [
        (0.65, "vs ランダム → 勝率65%達成"),
        (0.60, "vs 弱いDQN(snapshot) → 勝率60%達成"),
        (0.60, "vs 中程度のDQN(snapshot) → 勝率60%達成"),
    ]
    curriculum_phase = 0  # 0=vs random, 1以降=vs snapshot

    print_header()

    for episode in range(1, num_episodes + 1):
        stats = runner.run_episode()
        win_history.append(1 if stats["winner"] == Connect4Env.PLAYER1 else 0)
        reward_history.append(stats["reward_p1"])

        if episode % eval_interval == 0:
            recent     = win_history[-500:]
            win_rate   = np.mean(recent) * 100
            avg_reward = np.mean(reward_history[-500:])
            print(f"{episode:>8} | {opponent_label:>18} | {win_rate:>13.1f}% | {avg_reward:>10.3f} | {agent.epsilon:>7.5f}")

            # カリキュラム学習: しきい値を超えたら相手を更新
            if curriculum and curriculum_phase < len(CURRICULUM_PHASES):
                threshold, phase_desc = CURRICULUM_PHASES[curriculum_phase]
                if win_rate / 100 >= threshold:
                    snapshot_count += 1
                    snap_path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{snapshot_count:02d}_ep{episode}")
                    agent.save(snap_path)
                    print(f"\n  ✔ {phase_desc}")
                    print(f"  スナップショット保存: {snap_path}.npz")

                    curriculum_phase += 1

                    if curriculum_phase < len(CURRICULUM_PHASES):
                        # 新しい対戦相手: 今保存したスナップショット
                        new_opponent = DQNAgent()
                        new_opponent.load(snap_path + ".npz")
                        new_opponent.epsilon = 0.05  # 少しランダム性を残す
                        opponent       = new_opponent
                        opponent_label = f"snap{snapshot_count:02d}"
                        runner         = GameRunner(env, agent, opponent, renderer=None)
                        print(f"  対戦相手を更新: {opponent_label}\n")
                        print_header()
                    else:
                        print("  全フェーズ完了！\n")

    print("\n学習完了")
    agent.save(WEIGHTS_PATH)
    print(f"重みを保存しました: {WEIGHTS_PATH}.npz")

    # 勝率推移
    print("\n=== 勝率推移（直近500エピソード平均）===")
    window = 500
    for i in range(window, len(win_history) + 1, window):
        wr  = np.mean(win_history[i-window:i]) * 100
        bar = '#' * int(wr / 2)
        print(f"Ep {i:>6}: {bar:<50} {wr:.1f}%")

    sys.stdout = tee._stdout
    tee.close()

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",       type=int,  default=10000)
    parser.add_argument("--eval-interval",  type=int,  default=500)
    parser.add_argument("--opponent-path",  type=str,  default=None,
                        help="対戦相手のDQN重みパス（.npz不要）例: weights/snapshots/snapshot_01_ep5000")
    parser.add_argument("--curriculum",     action="store_true",
                        help="カリキュラム学習モード（勝率しきい値で相手を自動更新）")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        opponent_path=args.opponent_path,
        curriculum=args.curriculum,
    )
