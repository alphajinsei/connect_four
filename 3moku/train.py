"""
train.py — DQN エージェントの学習スクリプト（Connect Three: 5×5盤面、3目並べ）

設計方針:
  - DQN は常に PLAYER1（先手）として学習
  - 対戦相手は StochasticRuleBasedAgent（確率的ルールベース）
  - カリキュラムなし: 最初から強い相手と対戦して学習
  - 報酬は勝敗（±1.0）のみ。CNN が空間パターンを認識するため中間報酬は不要
  - **ランダム初期局面 + 確率的相手で訪問局面を広げる**（Stage 5 で導入）

Stage 5 の背景:
    学習済み DQN を決定論的 RuleBased と300回対戦させたところ、棋譜は2通り、
    DQN が遭遇する盤面はわずか11種類しかなかった（状態空間 ~6×10^6 の 0.0002%）。
    vs RuleBased 100% は「11局面の暗記テストで満点」を意味するに過ぎず、
    人間の変則手に対応できない構造的原因だった。

    実測した対策効果（500ゲームで DQN が遭遇する異なる盤面数）:
        ベースライン（決定論 RB, 開幕なし）    :  11
        ランダム初期局面 k<=4 のみ             : 413
        確率的 RB (noise=0.15) のみ            : 108
        両方                                   : 596

使い方:
    # ゼロから学習
    .venv/Scripts/python 3moku/train.py --episodes 30000

    # 学習済み重みから続き
    .venv/Scripts/python 3moku/train.py --load-path weights/dqn_connect3 --episodes 30000

    # Stage 4 以前の挙動を再現（比較用）
    .venv/Scripts/python 3moku/train.py --opening-plies 0 --opponent-noise 0.0
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
from agents.stochastic_rule_based_agent import StochasticRuleBasedAgent
from agents.random_agent import RandomAgent
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


def eval_vs(agent, env, opponent, n=200, opening_plies=0):
    """
    ε=0 の純粋推論で opponent と n 戦し、勝率(%)と訪問局面集合を返す。

    opening_plies > 0 なら毎回ランダム初期局面から開始する。これが汎化性能の
    本命指標: 決定論的な開幕からの勝率は「暗記した1本道」の再現に過ぎない。
    """
    runner        = GameRunner(env, agent, opponent, renderer=None)
    saved_eps     = agent.epsilon
    agent.epsilon = 0.0
    wins  = 0
    seen  = set()
    for _ in range(n):
        stats = runner.run_episode(random_opening_plies=opening_plies)
        if stats["winner"] == Connect3Env.PLAYER1:
            wins += 1
        seen.update(stats["p1_states"])
    agent.epsilon = saved_eps
    return wins / n * 100, seen


def print_header(opening_plies, opponent_noise):
    print(f"\n=== vs 確率的RuleBased 直接対戦学習 "
          f"(開幕ランダム≤{opening_plies}手, 相手noise={opponent_noise}) ===")
    print(f"{'Episode':>8} | {'勝率(直近1000)':>13} | {'平均報酬':>9} | {'ε':>7} | "
          f"{'vs RB(固定)':>11} | {'vs RB(乱開幕)':>13} | {'vs Random':>9} | {'訪問局面':>8}")
    print("-" * 115)


def train(num_episodes=30000, eval_interval=500, load_path=None, no_buffer=False,
          opening_plies=4, opponent_noise=0.15):
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
    print(f"対戦相手: StochasticRuleBasedAgent(noise={opponent_noise})")
    print(f"開幕ランダム: 0〜{opening_plies}手（偶数手のみ、DQNは先手を維持）")
    print()

    win_history    = []
    reward_history = []
    best_vs_open   = 0.0   # 主指標: ランダム開幕からの vs RuleBased 勝率
    train_states   = set() # 学習中に DQN が遭遇した異なる盤面（累積）

    opp = StochasticRuleBasedAgent(noise=opponent_noise)
    print_header(opening_plies, opponent_noise)

    for episode in range(1, num_episodes + 1):
        stats = GameRunner(env, agent, opp, renderer=None).run_episode(
            random_opening_plies=opening_plies
        )
        win_history.append(1 if stats["winner"] == Connect3Env.PLAYER1 else 0)
        reward_history.append(stats["reward_p1"])
        train_states.update(stats["p1_states"])

        if episode % eval_interval == 0:
            win_rate   = np.mean(win_history[-1000:]) * 100
            avg_reward = np.mean(reward_history[-1000:])
            # 従来指標（Stage 4 との比較用。決定論的な1本道なので過大評価される）
            vs_rb,   _ = eval_vs(agent, env, RuleBasedAgent(), n=EVAL_N)
            # 主指標: ランダム開幕からの勝率 = 汎化性能
            vs_open, _ = eval_vs(agent, env, RuleBasedAgent(), n=EVAL_N,
                                 opening_plies=opening_plies)
            vs_rand, _ = eval_vs(agent, env, RandomAgent(), n=EVAL_N)

            print(f"{episode:>8} | {win_rate:>12.1f}% | {avg_reward:>9.3f} | {agent.epsilon:>7.5f} | "
                  f"{vs_rb:>10.1f}% | {vs_open:>12.1f}% | {vs_rand:>8.1f}% | {len(train_states):>8}")

            # 主指標（ランダム開幕勝率）がベスト更新時にスナップショット保存
            if vs_open > best_vs_open:
                best_vs_open = vs_open
                snap_path = os.path.join(
                    SNAPSHOTS_DIR,
                    f"ep{episode}_open{vs_open:.0f}_rb{vs_rb:.0f}_rand{vs_rand:.0f}pct_{session_ts}"
                )
                agent.save(snap_path)
                print(f"  [Snap] 乱開幕 {vs_open:.1f}% / 固定RB {vs_rb:.1f}% / Rand {vs_rand:.1f}% → {snap_path}.pt")

    print("\n学習完了")
    agent.save_checkpoint(WEIGHTS_PATH)
    print(f"重み+チェックポイントを保存: {WEIGHTS_PATH}.pt / {WEIGHTS_PATH}_checkpoint.pt")
    print(f"vs RuleBased(ランダム開幕) ベスト: {best_vs_open:.1f}%")
    print(f"学習中に遭遇した異なる盤面: {len(train_states)}（Stage 4 は約11）")

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
    parser.add_argument("--opening-plies", type=int, default=4,
                        help="開幕ランダム手数の上限（0でStage4以前の挙動）")
    parser.add_argument("--opponent-noise", type=float, default=0.15,
                        help="相手のランダム手混入率（0.0で決定論的RuleBased）")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        load_path=args.load_path,
        no_buffer=args.no_buffer,
        opening_plies=args.opening_plies,
        opponent_noise=args.opponent_noise,
    )
