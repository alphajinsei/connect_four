"""
train.py — DQN エージェントの学習スクリプト（Connect Three: 5×5盤面、3目並べ）

設計方針:
  - DQN は常に PLAYER1（先手）として学習
  - 対戦相手は RuleBasedAgent（完全ルールベース）
  - カリキュラムなし: 最初から強い相手と対戦して学習
  - 報酬は勝敗（±1.0）のみ。CNN が空間パターンを認識するため中間報酬は不要

使い方:
    # ゼロから学習
    .venv/Scripts/python 3moku/train.py --episodes 30000

    # 学習済み重みから続き
    .venv/Scripts/python 3moku/train.py --load-path weights/dqn_connect3 --episodes 30000
"""
import sys
import os
import argparse
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from env.connect3_env import Connect3Env
from agents.dqn_agent import DQNAgent
from agents.rule_based_agent import RuleBasedAgent
from game_runner import GameRunner

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH  = os.path.join(_SCRIPT_DIR, "weights", "dqn_connect3")
SNAPSHOTS_DIR = os.path.join(_SCRIPT_DIR, "weights", "snapshots")
LOG_PATH      = os.path.join(_SCRIPT_DIR, "weights", "train_log.txt")

EVAL_N = 200


class Tee:
    """stdout とファイルに同時に書き出す。"""
    def __init__(self, path):
        self._path   = path
        self._stdout = sys.stdout
        with open(path, "w", encoding="utf-8") as f:
            pass

    def write(self, data):
        self._stdout.write(data)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

    def flush(self):
        self._stdout.flush()

    def close(self):
        pass


def make_agent(**kwargs):
    defaults = dict(
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.10,
        epsilon_decay=0.99990,
        buffer_capacity=20000,
        batch_size=128,
        warmup_steps=2000,
        target_update_interval=500,
    )
    defaults.update(kwargs)
    return DQNAgent(**defaults)


def eval_vs_rulebased(agent, env, n=200):
    runner        = GameRunner(env, agent, RuleBasedAgent(), renderer=None)
    saved_eps     = agent.epsilon
    agent.epsilon = 0.0
    wins = sum(
        1 for _ in range(n)
        if runner.run_episode()["winner"] == Connect3Env.PLAYER1
    )
    agent.epsilon = saved_eps
    return wins / n * 100


def print_header():
    print(f"\n=== vs RuleBased 直接対戦学習 ===")
    print(f"{'Episode':>8} | {'勝率(直近1000)':>15} | {'平均報酬':>10} | {'ε':>7} | {'vs RuleBased(200)':>18}")
    print("-" * 72)


def train(num_episodes=30000, eval_interval=500, load_path=None, no_buffer=False):
    os.makedirs(os.path.join(_SCRIPT_DIR, "weights"), exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if os.path.exists(LOG_PATH):
        import shutil
        backup = LOG_PATH.replace(".txt", f"_{session_ts}.txt")
        try:
            os.rename(LOG_PATH, backup)
        except PermissionError:
            shutil.copy2(LOG_PATH, backup)
        print(f"前回ログを退避: {backup}", file=sys.stderr)

    tee        = Tee(LOG_PATH)
    sys.stdout = tee

    env = Connect3Env()

    if load_path:
        agent = make_agent(epsilon_start=0.15)
        ckpt_path = load_path + '_checkpoint.pt'
        if os.path.exists(ckpt_path) and not no_buffer:
            agent.load_checkpoint(load_path, load_buffer=True)
            print(f"チェックポイントをロード: {load_path}  (ε={agent.epsilon:.4f}, steps={agent.total_steps}, buffer={len(agent.replay_buffer)})")
        elif os.path.exists(ckpt_path):
            agent.load_checkpoint(load_path, load_buffer=False)
            print(f"重み+状態をロード(バッファなし): {load_path}  (ε={agent.epsilon:.4f}, steps={agent.total_steps})")
        else:
            agent.load(load_path + ".pt")
            print(f"重みをロード: {load_path}.pt  (ε={agent.epsilon:.4f})")
    else:
        agent = make_agent()
        print("新規学習開始（Connect Three: 5×5盤面、3目並べ、CNN + vs RuleBased直接対戦 + PyTorch）")

    print(f"ハイパーパラメータ: lr=5e-4, epsilon_end=0.10, target_update=500, buffer=20000")
    print(f"対戦相手: RuleBasedAgent（完全ルールベース）")
    print()

    win_history    = []
    reward_history = []
    best_vs_rb     = 0.0

    opp = RuleBasedAgent()
    print_header()

    for episode in range(1, num_episodes + 1):
        stats = GameRunner(env, agent, opp, renderer=None).run_episode()
        win_history.append(1 if stats["winner"] == Connect3Env.PLAYER1 else 0)
        reward_history.append(stats["reward_p1"])

        if episode % eval_interval == 0:
            win_rate   = np.mean(win_history[-1000:]) * 100
            avg_reward = np.mean(reward_history[-1000:])
            vs_rb      = eval_vs_rulebased(agent, env, n=EVAL_N)

            print(f"{episode:>8} | {win_rate:>13.1f}% | {avg_reward:>10.3f} | {agent.epsilon:>7.5f} | {vs_rb:>17.1f}%")

            if vs_rb > best_vs_rb:
                best_vs_rb = vs_rb
                best_path  = os.path.join(SNAPSHOTS_DIR, f"best_ep{episode}_rb{vs_rb:.0f}pct_{session_ts}")
                agent.save(best_path)
                print(f"  [Best] vs RuleBased {vs_rb:.1f}% → {best_path}.pt")

    print("\n学習完了")
    agent.save_checkpoint(WEIGHTS_PATH)
    print(f"重み+チェックポイントを保存: {WEIGHTS_PATH}.pt / {WEIGHTS_PATH}_checkpoint.pt")
    print(f"vs RuleBased ベスト: {best_vs_rb:.1f}%")

    sys.stdout = tee._stdout
    tee.close()
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",      type=int, default=30000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--load-path",     type=str, default=None,
                        help="学習済み重みから再開 例: weights/dqn_connect3")
    parser.add_argument("--no-buffer",     action="store_true",
                        help="ロード時にReplayBufferを引き継がない")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        load_path=args.load_path,
        no_buffer=args.no_buffer,
    )
