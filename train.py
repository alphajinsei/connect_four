"""
train.py — DQN エージェントの学習スクリプト（ステージ5: カリキュラム学習）

設計方針:
  - DQN は常に PLAYER1（先手）として学習
  - 対戦相手は NoisyRuleBasedAgent（noise で強さを制御）
  - 勝率しきい値を超えたら自動的に次のフェーズへ移行
  - 攻防セット中間報酬（connect4_env.py 内）

カリキュラム:
  Phase 1: noise=0.8 → 目標勝率 80%
  Phase 2: noise=0.5 → 目標勝率 70%
  Phase 3: noise=0.2 → 目標勝率 50%
  Phase 4: noise=0.0 → 目標勝率 50%（最終目標）

使い方:
    # ゼロから学習
    .venv/Scripts/python train.py --episodes 30000

    # 学習済み重みから続き
    .venv/Scripts/python train.py --load-path weights/dqn_connect4 --episodes 30000
"""
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from env.connect4_env import Connect4Env
from agents.dqn_agent import DQNAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.noisy_rule_based_agent import NoisyRuleBasedAgent
from game_runner import GameRunner

WEIGHTS_PATH  = "weights/dqn_connect4"
SNAPSHOTS_DIR = "weights/snapshots"
LOG_PATH      = "weights/train_log.txt"

# カリキュラム定義: (noise, 目標勝率%)
CURRICULUM = [
    (0.8, 80.0),
    (0.5, 70.0),
    (0.2, 50.0),
    (0.0, 50.0),
]


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
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.10,
        epsilon_decay=0.99990,
        buffer_capacity=50000,
        batch_size=128,
        warmup_steps=2000,
        target_update_interval=500,
    )
    defaults.update(kwargs)
    return DQNAgent(**defaults)


def eval_vs_rulebased(agent, env, n=200):
    """ルールベースAI（noise=0）と n 戦して勝率を返す（greedy、学習なし）"""
    runner        = GameRunner(env, agent, RuleBasedAgent(), renderer=None)
    saved_eps     = agent.epsilon
    agent.epsilon = 0.0
    wins = sum(
        1 for _ in range(n)
        if runner.run_episode()["winner"] == Connect4Env.PLAYER1
    )
    agent.epsilon = saved_eps
    return wins / n * 100


def eval_vs_noisy(agent, env, noise, n=200):
    """NoisyRuleBasedAgent（指定 noise）と n 戦して勝率を返す（greedy）"""
    runner        = GameRunner(env, agent, NoisyRuleBasedAgent(noise=noise), renderer=None)
    saved_eps     = agent.epsilon
    agent.epsilon = 0.0
    wins = sum(
        1 for _ in range(n)
        if runner.run_episode()["winner"] == Connect4Env.PLAYER1
    )
    agent.epsilon = saved_eps
    return wins / n * 100


def print_header(phase, noise, target):
    print(f"\n=== Phase {phase}: noise={noise:.1f}, 目標勝率={target:.0f}% ===")
    print(f"{'Episode':>8} | {'勝率(直近500)':>14} | {'平均報酬':>10} | {'ε':>7} | {'vs Noisy(200)':>14} | {'vs RuleBased(200)':>18}")
    print("-" * 82)


def train(num_episodes=30000, eval_interval=500, load_path=None):
    os.makedirs("weights",     exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    tee        = Tee(LOG_PATH)
    sys.stdout = tee

    env = Connect4Env()

    if load_path:
        agent = make_agent(epsilon_start=0.15)
        agent.load(load_path + ".npz")
        print(f"重みをロード: {load_path}.npz  (ε={agent.epsilon:.4f})")
    else:
        agent = make_agent()
        print("新規学習開始（ステージ5: カリキュラム学習）")

    print(f"ハイパーパラメータ: lr=5e-4, epsilon_end=0.10, target_update=500")
    print(f"カリキュラム: {CURRICULUM}")
    print()

    win_history    = []
    reward_history = []
    best_vs_rb     = 0.0

    phase_idx      = 0
    noise, target  = CURRICULUM[phase_idx]
    opp            = NoisyRuleBasedAgent(noise=noise)
    print_header(phase_idx + 1, noise, target)

    for episode in range(1, num_episodes + 1):
        stats = GameRunner(env, agent, opp, renderer=None).run_episode()
        win_history.append(1 if stats["winner"] == Connect4Env.PLAYER1 else 0)
        reward_history.append(stats["reward_p1"])

        if episode % eval_interval == 0:
            win_rate   = np.mean(win_history[-500:]) * 100
            avg_reward = np.mean(reward_history[-500:])
            vs_noisy   = eval_vs_noisy(agent, env, noise)
            vs_rb      = eval_vs_rulebased(agent, env)

            print(f"{episode:>8} | {win_rate:>13.1f}% | {avg_reward:>10.3f} | {agent.epsilon:>7.5f} | {vs_noisy:>13.1f}% | {vs_rb:>17.1f}%")

            # 自己ベスト更新時にスナップショット保存
            if vs_rb > best_vs_rb:
                best_vs_rb = vs_rb
                best_path  = os.path.join(SNAPSHOTS_DIR, f"best_ep{episode}_rb{vs_rb:.0f}pct")
                agent.save(best_path)
                print(f"  [Best] vs RuleBased {vs_rb:.1f}% → {best_path}.npz")

            # カリキュラム移行チェック
            if vs_noisy >= target and phase_idx < len(CURRICULUM) - 1:
                phase_idx += 1
                noise, target = CURRICULUM[phase_idx]
                opp = NoisyRuleBasedAgent(noise=noise)
                print(f"\n  [Phase Up] → Phase {phase_idx + 1}: noise={noise:.1f}, 目標勝率={target:.0f}%")
                print_header(phase_idx + 1, noise, target)
                win_history.clear()
                reward_history.clear()

    print("\n学習完了")
    agent.save(WEIGHTS_PATH)
    print(f"重みを保存しました: {WEIGHTS_PATH}.npz")
    print(f"最終フェーズ: Phase {phase_idx + 1} (noise={noise:.1f})")
    print(f"vs RuleBased ベスト: {best_vs_rb:.1f}%")

    sys.stdout = tee._stdout
    tee.close()
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",      type=int, default=30000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--load-path",     type=str, default=None,
                        help="学習済み重みから再開 例: weights/dqn_connect4")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        load_path=args.load_path,
    )
