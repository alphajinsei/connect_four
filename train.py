"""
train.py — DQN エージェントの学習スクリプト（ステージ5: カリキュラム学習）

設計方針:
  - DQN は常に PLAYER1（先手）として学習
  - 対戦相手は NoisyRuleBasedAgent（noise で強さを制御）
  - 勝率しきい値を超えたら自動的に次のフェーズへ移行
  - 攻防セット中間報酬（connect4_env.py 内）

カリキュラム:
  Phase 1: noise=0.8 → 目標勝率 80%（vs Noisy 500戦、2回連続クリアで昇格）
  Phase 2: noise=0.5 → 目標勝率 70%（vs Noisy 500戦、2回連続クリアで昇格）
  Phase 3: noise=0.2 → 目標勝率 50%（vs Noisy 500戦、2回連続クリアで昇格）
  Phase 4: noise=0.1（NoisyRuleBased）50% + スナップショットプール 50% → 目標勝率 50%

昇格基準の設計思想:
  - 200戦では統計的振れ幅が大きく（±7%程度）まぐれ昇格が起きた（ph3を1500epで通過）
  - 500戦 + 2回連続クリアにすることで、一時的な上振れによる早期昇格を防ぐ

Phase 4の設計思想:
  - 完全ルールベース固定では「固定相手への過学習→崩壊」が繰り返された（ステージ4の失敗再現）
  - noise=0.1（90%ルールベース、10%ランダム）で微妙なランダム性を加えて癖への過学習を防ぐ
  - スナップショット（過去の自分）を50%混ぜて多様性を確保し、破滅的忘却を防ぐ

使い方:
    # ゼロから学習
    .venv/Scripts/python train.py --episodes 30000

    # 学習済み重みから続き
    .venv/Scripts/python train.py --load-path weights/dqn_connect4 --episodes 30000

    # フェーズ指定で再開
    .venv/Scripts/python train.py --load-path weights/snapshots/best_epXXXX --start-phase 3
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
    (0.1, 50.0),  # Phase 4: noise=0.1 + スナップショットプール
]

# 昇格に必要な連続クリア回数
PHASE_UP_CONSECUTIVE = 2
# 昇格判定のeval対戦数（振れ幅を減らすため200→500）
PHASE_UP_EVAL_N = 500


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


def load_snapshot_pool(snapshot_dir, min_rb_pct=50):
    """スナップショットからvs RuleBased勝率がmin_rb_pct以上のものだけ読み込む"""
    import re
    pool = []
    for fname in os.listdir(snapshot_dir):
        if not fname.endswith(".npz"):
            continue
        m = re.search(r"rb(\d+)pct", fname)
        if m and int(m.group(1)) >= min_rb_pct:
            a = make_agent(epsilon_start=0.0)
            a.load(os.path.join(snapshot_dir, fname))
            a.epsilon = 0.0
            pool.append(a)
    return pool


def make_phase4_opponent(snapshot_pool):
    """Phase 4用: 50%でNoisyRuleBased(noise=0.1)、50%でスナップショットからランダム選択"""
    if snapshot_pool and np.random.random() < 0.5:
        return np.random.choice(snapshot_pool)
    return NoisyRuleBasedAgent(noise=0.1)


def print_header(phase, noise, target, is_phase4=False):
    if is_phase4:
        desc = f"noise=0.1×50% + snapshot×50%"
    else:
        desc = f"noise={noise:.1f}"
    print(f"\n=== Phase {phase}: {desc}, 目標勝率={target:.0f}% (昇格条件: {PHASE_UP_EVAL_N}戦×{PHASE_UP_CONSECUTIVE}回連続) ===")
    print(f"{'Episode':>8} | {'勝率(直近500)':>14} | {'平均報酬':>10} | {'ε':>7} | {'vs Noisy(500)':>14} | {'vs RuleBased(200)':>18}")
    print("-" * 88)


def train(num_episodes=30000, eval_interval=500, load_path=None, start_phase=1, max_phase=None, no_buffer=False):
    os.makedirs("weights",     exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    # 既存ログをフェーズ番号+タイムスタンプ付きでバックアップ
    if os.path.exists(LOG_PATH):
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = LOG_PATH.replace(".txt", f"_ph{start_phase - 1}_{ts}.txt")
        os.rename(LOG_PATH, backup)
        print(f"前回ログを退避: {backup}", file=sys.stderr)

    tee        = Tee(LOG_PATH)
    sys.stdout = tee

    env = Connect4Env()

    if load_path:
        agent = make_agent(epsilon_start=0.15)  # フォールバック用ε
        ckpt_path = load_path + '_checkpoint.npz'
        if os.path.exists(ckpt_path) and not no_buffer:
            agent.load_checkpoint(load_path, load_buffer=True)
            print(f"チェックポイントをロード: {load_path}  (ε={agent.epsilon:.4f}, steps={agent.total_steps}, buffer={len(agent.replay_buffer)})")
        elif os.path.exists(ckpt_path):
            agent.load_checkpoint(load_path, load_buffer=False)
            print(f"重み+状態をロード(バッファなし): {load_path}  (ε={agent.epsilon:.4f}, steps={agent.total_steps})")
        else:
            agent.load(load_path + ".npz")
            print(f"重みをロード: {load_path}.npz  (ε={agent.epsilon:.4f})")
    else:
        agent = make_agent()
        print("新規学習開始（ステージ6: カリキュラム学習）")

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
    is_phase4   = (phase_idx == len(CURRICULUM) - 1)

    # Phase 4用スナップショットプール
    snapshot_pool = load_snapshot_pool(SNAPSHOTS_DIR) if is_phase4 else []
    if is_phase4:
        print(f"スナップショットプール: {len(snapshot_pool)}体ロード")

    opp = make_phase4_opponent(snapshot_pool) if is_phase4 else NoisyRuleBasedAgent(noise=noise)
    print_header(phase_idx + 1, noise, target, is_phase4)

    for episode in range(1, num_episodes + 1):
        # Phase 4はエピソードごとに対戦相手をランダム選択
        if is_phase4:
            opp = make_phase4_opponent(snapshot_pool)

        stats = GameRunner(env, agent, opp, renderer=None).run_episode()
        win_history.append(1 if stats["winner"] == Connect4Env.PLAYER1 else 0)
        reward_history.append(stats["reward_p1"])

        if episode % eval_interval == 0:
            win_rate   = np.mean(win_history[-500:]) * 100
            avg_reward = np.mean(reward_history[-500:])
            vs_noisy   = eval_vs_noisy(agent, env, noise, n=PHASE_UP_EVAL_N)
            vs_rb      = eval_vs_rulebased(agent, env)

            print(f"{episode:>8} | {win_rate:>13.1f}% | {avg_reward:>10.3f} | {agent.epsilon:>7.5f} | {vs_noisy:>13.1f}% | {vs_rb:>17.1f}%")

            # 自己ベスト更新時にスナップショット保存
            if vs_rb > best_vs_rb:
                best_vs_rb = vs_rb
                best_path  = os.path.join(SNAPSHOTS_DIR, f"best_ep{episode}_rb{vs_rb:.0f}pct")
                agent.save(best_path)
                print(f"  [Best] vs RuleBased {vs_rb:.1f}% → {best_path}.npz")
                # Phase 4ならプールに追加
                if is_phase4:
                    new_agent = make_agent(epsilon_start=0.0)
                    new_agent.load(best_path + ".npz")
                    new_agent.epsilon = 0.0
                    snapshot_pool.append(new_agent)
                    print(f"  [Pool] スナップショットプールに追加（計{len(snapshot_pool)}体）")

            # カリキュラム移行チェック（2回連続クリアで昇格）
            if not is_phase4 and vs_noisy >= target and (max_phase is None or phase_idx + 1 < max_phase):
                consecutive_clears += 1
                print(f"  [Clear {consecutive_clears}/{PHASE_UP_CONSECUTIVE}] vs Noisy {vs_noisy:.1f}% >= {target:.0f}%")
                if consecutive_clears >= PHASE_UP_CONSECUTIVE:
                    # 昇格時スナップショット（フェーズ卒業時点の重み）
                    grad_path = os.path.join(SNAPSHOTS_DIR, f"phaseup_ph{phase_idx + 1}_ep{episode}_rb{vs_rb:.0f}pct")
                    agent.save(grad_path)
                    print(f"  [PhaseUp Snapshot] {grad_path}.npz")

                    phase_idx += 1
                    noise, target = CURRICULUM[phase_idx]
                    is_phase4 = (phase_idx == len(CURRICULUM) - 1)
                    consecutive_clears = 0
                    win_history.clear()
                    reward_history.clear()
                    if is_phase4:
                        snapshot_pool = load_snapshot_pool(SNAPSHOTS_DIR)
                        opp = make_phase4_opponent(snapshot_pool)
                        print(f"\n  [Phase Up] → Phase {phase_idx + 1}: noise=0.1×50% + snapshot×50%（{len(snapshot_pool)}体）, 目標勝率={target:.0f}%")
                    else:
                        opp = NoisyRuleBasedAgent(noise=noise)
                        print(f"\n  [Phase Up] → Phase {phase_idx + 1}: noise={noise:.1f}, 目標勝率={target:.0f}%")
                    print_header(phase_idx + 1, noise, target, is_phase4)
            else:
                consecutive_clears = 0

    print("\n学習完了")
    agent.save_checkpoint(WEIGHTS_PATH)
    print(f"重み+チェックポイントを保存: {WEIGHTS_PATH}.npz / {WEIGHTS_PATH}_checkpoint.npz")
    print(f"最終フェーズ: Phase {phase_idx + 1}")
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
    parser.add_argument("--start-phase",   type=int, default=1,
                        help="開始フェーズ (1-4, デフォルト: 1)")
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
