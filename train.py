"""
train.py — DQN エージェントの学習スクリプト（ステージ8: カリキュラム学習 + 中間報酬なし）

設計方針:
  - DQN は常に PLAYER1（先手）として学習
  - 対戦相手は NoisyRuleBasedAgent（noise で強さを制御）
  - 勝率しきい値を超えたら自動的に次のフェーズへ移行
  - 報酬は勝敗（±1.0）のみ、中間報酬なし

カリキュラム:
  Phase 1: noise=0.8 → 目標勝率 75%（vs Noisy 200戦、2回連続クリアで昇格）
  Phase 2: noise=0.5 → 目標勝率 70%（vs Noisy 200戦、2回連続クリアで昇格）
  Phase 3: noise=0.2 → 目標勝率 60%（vs Noisy 200戦、2回連続クリアで昇格）
  Phase 4: noise=0.1 → 目標勝率 50%（vs Noisy 200戦、2回連続クリアで昇格）
  Phase 5: noise=0.0 → 目標勝率 40%（完全ルールベースAI）

昇格基準の設計思想:
  - 200戦 + 2回連続クリアで、一時的な上振れによる早期昇格を防ぐ

使い方:
    # ゼロから学習
    .venv/Scripts/python train.py --episodes 50000

    # 学習済み重みから続き
    .venv/Scripts/python train.py --load-path weights/dqn_connect4 --episodes 50000

    # フェーズ指定で再開
    .venv/Scripts/python train.py --load-path weights/snapshots/best_epXXXX --start-phase 3
"""
import sys
import os
import argparse
from datetime import datetime
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
    (0.8, 75.0),   # Phase 1: ほぼランダム
    (0.5, 70.0),   # Phase 2: 半分ルールベース
    (0.2, 60.0),   # Phase 3: 8割ルールベース
    (0.1, 50.0),   # Phase 4: 9割ルールベース
    (0.0, 40.0),   # Phase 5: 完全ルールベースAI
]

# 昇格に必要な連続クリア回数
PHASE_UP_CONSECUTIVE = 2
# 昇格判定のeval対戦数（2回連続クリアで精度を担保）
PHASE_UP_EVAL_N = 200


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
        buffer_capacity=200000,
        batch_size=128,
        warmup_steps=2000,
        target_update_interval=500,
    )
    defaults.update(kwargs)
    return DQNAgent(**defaults)


def eval_vs_rulebased(agent, env, n=100):
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
    desc = f"noise={noise:.1f}" if noise > 0 else "RuleBased(純粋)"
    print(f"\n=== Phase {phase}: {desc}, 目標勝率={target:.0f}% (昇格条件: {PHASE_UP_EVAL_N}戦×{PHASE_UP_CONSECUTIVE}回連続) ===")
    print(f"{'Episode':>8} | {'勝率(直近1000)':>15} | {'平均報酬':>10} | {'ε':>7} | {'vs Noisy(200)':>14} | {'vs RuleBased(100)':>18}")
    print("-" * 88)


def train(num_episodes=30000, eval_interval=500, load_path=None, start_phase=1, max_phase=None, no_buffer=False):
    os.makedirs("weights",     exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    # 学習セッションのタイムスタンプ（スナップショット名に使用）
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 既存ログをフェーズ番号+タイムスタンプ付きでバックアップ
    if os.path.exists(LOG_PATH):
        backup = LOG_PATH.replace(".txt", f"_ph{start_phase - 1}_{session_ts}.txt")
        os.rename(LOG_PATH, backup)
        print(f"前回ログを退避: {backup}", file=sys.stderr)

    tee        = Tee(LOG_PATH)
    sys.stdout = tee

    env = Connect4Env()

    if load_path:
        agent = make_agent(epsilon_start=0.15)  # フォールバック用ε
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
        print("新規学習開始（ステージ8: カリキュラム学習 + 中間報酬なし + PyTorch）")

    print(f"ハイパーパラメータ: lr=5e-4, epsilon_end=0.10, target_update=500")
    print(f"カリキュラム: {CURRICULUM}")
    print(f"昇格条件: vs Noisy {PHASE_UP_EVAL_N}戦 × {PHASE_UP_CONSECUTIVE}回連続クリア")
    print()

    win_history    = []
    reward_history = []
    best_vs_rb     = 0.0
    consecutive_clears = 0  # 連続クリア回数

    phase_idx   = max(0, start_phase - 1)
    noise, target = CURRICULUM[phase_idx]

    opp = NoisyRuleBasedAgent(noise=noise) if noise > 0 else RuleBasedAgent()
    print_header(phase_idx + 1, noise, target)

    for episode in range(1, num_episodes + 1):
        stats = GameRunner(env, agent, opp, renderer=None).run_episode()
        win_history.append(1 if stats["winner"] == Connect4Env.PLAYER1 else 0)
        reward_history.append(stats["reward_p1"])

        if episode % eval_interval == 0:
            win_rate   = np.mean(win_history[-1000:]) * 100
            avg_reward = np.mean(reward_history[-1000:])
            vs_noisy   = eval_vs_noisy(agent, env, noise, n=PHASE_UP_EVAL_N)
            vs_rb      = eval_vs_rulebased(agent, env)

            print(f"{episode:>8} | {win_rate:>13.1f}% | {avg_reward:>10.3f} | {agent.epsilon:>7.5f} | {vs_noisy:>13.1f}% | {vs_rb:>17.1f}%")

            # 自己ベスト更新時にスナップショット保存
            if vs_rb > best_vs_rb:
                best_vs_rb = vs_rb
                best_path  = os.path.join(SNAPSHOTS_DIR, f"best_ep{episode}_rb{vs_rb:.0f}pct_{session_ts}")
                agent.save(best_path)
                print(f"  [Best] vs RuleBased {vs_rb:.1f}% → {best_path}.pt")

            # カリキュラム移行チェック（2回連続クリアで昇格）
            is_last_phase = (phase_idx >= len(CURRICULUM) - 1)
            if not is_last_phase and vs_noisy >= target and (max_phase is None or phase_idx + 1 < max_phase):
                consecutive_clears += 1
                print(f"  [Clear {consecutive_clears}/{PHASE_UP_CONSECUTIVE}] vs Noisy {vs_noisy:.1f}% >= {target:.0f}%")
                if consecutive_clears >= PHASE_UP_CONSECUTIVE:
                    # 昇格時スナップショット（フェーズ卒業時点の重み）
                    grad_path = os.path.join(SNAPSHOTS_DIR, f"phaseup_ph{phase_idx + 1}_ep{episode}_rb{vs_rb:.0f}pct_{session_ts}")
                    agent.save(grad_path)
                    print(f"  [PhaseUp Snapshot] {grad_path}.pt")

                    phase_idx += 1
                    noise, target = CURRICULUM[phase_idx]
                    consecutive_clears = 0
                    win_history.clear()
                    reward_history.clear()
                    opp = NoisyRuleBasedAgent(noise=noise) if noise > 0 else RuleBasedAgent()
                    desc = f"noise={noise:.1f}" if noise > 0 else "RuleBased(純粋)"
                    print(f"\n  [Phase Up] → Phase {phase_idx + 1}: {desc}, 目標勝率={target:.0f}%")
                    print_header(phase_idx + 1, noise, target)
            else:
                consecutive_clears = 0

    print("\n学習完了")
    agent.save_checkpoint(WEIGHTS_PATH)
    print(f"重み+チェックポイントを保存: {WEIGHTS_PATH}.pt / {WEIGHTS_PATH}_checkpoint.pt")
    print(f"最終フェーズ: Phase {phase_idx + 1}")
    print(f"vs RuleBased ベスト: {best_vs_rb:.1f}%")

    sys.stdout = tee._stdout
    tee.close()
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",      type=int, default=30000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--load-path",     type=str, default=None,
                        help="学習済み重みから再開 例: weights/dqn_connect4")
    parser.add_argument("--start-phase",   type=int, default=1,
                        help="開始フェーズ (1-5, デフォルト: 1)")
    parser.add_argument("--max-phase",     type=int, default=None,
                        help="このフェーズ以上には移行しない（例: --max-phase 3 でph3固定）")
    parser.add_argument("--no-buffer",     action="store_true",
                        help="ロード時にReplayBufferを引き継がない（重み+学習状態のみ復元）")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        load_path=args.load_path,
        start_phase=args.start_phase,
        max_phase=args.max_phase,
        no_buffer=args.no_buffer,
    )
