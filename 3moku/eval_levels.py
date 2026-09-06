"""
eval_levels.py — Web公開版で対戦相手として選べる各レベルの強さを実測する

なぜ必要か:
    サイトに「勝率○%」と表示する以上、その数字には裏取りが要る。
    train.py の学習ログにも勝率は出るが、あれは 200戦での測定であり、
    このプロジェクトでは「200戦の99%が400戦では81.8%だった」という
    サンプル数不足の失敗を既に踏んでいる（CLAUDE.md 参照）。
    表示用の数字はここで N=400 以上で測り直す。

使い方:
    .venv/Scripts/python 3moku/eval_levels.py
    .venv/Scripts/python 3moku/eval_levels.py --n 1000

実測結果（2026-09-06, n=400）:
    レベル              | vs Random | vs 確率的RB | vs RB(乱開幕)
    Lv.0 学習前(ランダム) |    64.2%  |     6.0%   |     9.2%
    Lv.1 学習1,000回     |    95.2%  |    92.5%   |    73.8%
    Lv.2 学習4,000回     |    95.2%  |    91.2%   |    79.8%
    Lv.3 学習20,000回    |    99.0%  |    99.0%   |    89.0%

    → web-onnx/index.html には「vs RB(乱開幕)」を表示している。
      vs Random は Lv.1 と Lv.2 がどちらも 95.2% で飽和しており、
      レベル間の差が見えないため指標として使えない。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from agents.stochastic_rule_based_agent import StochasticRuleBasedAgent
from env.connect3_env import Connect3Env
from train import eval_vs

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SNAP = os.path.join(_SCRIPT_DIR, "weights", "snapshots")

# Web版のレベル定義と対応する重み。web-onnx/index.html の LEVELS と同じ並び。
LEVELS = [
    ("Lv.1 学習1,000回",  os.path.join(_SNAP, "ep1000_open76_rb72_rand92pct_20260816_124656.pt")),
    ("Lv.2 学習4,000回",  os.path.join(_SNAP, "ep4000_open90_rb100_rand95pct_20260816_124656.pt")),
    ("Lv.3 学習20,000回", os.path.join(_SCRIPT_DIR, "weights", "dqn_connect3.pt")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="1条件あたりの対戦数（既定400）")
    args = ap.parse_args()
    n = args.n

    env = Connect3Env()

    print(f"各レベルの強さ（1条件あたり {n} 戦, ε=0）\n")
    print(f"{'レベル':<20} | {'vs Random':>10} | {'vs 確率的RB':>12} | "
          f"{'vs RB(乱開幕)':>14} | {'訪問局面':>8}")
    print("-" * 82)

    # Lv.0（ランダム）は学習なしの基準点。DQNではないので RandomAgent 同士で測る。
    rnd = RandomAgent()
    runner_env = Connect3Env()
    w_rand, _ = eval_vs(_Wrap(rnd), runner_env, RandomAgent(), n=n)
    w_rb, _ = eval_vs(_Wrap(rnd), runner_env, StochasticRuleBasedAgent(noise=0.15), n=n)
    w_open, seen0 = eval_vs(_Wrap(rnd), runner_env, StochasticRuleBasedAgent(noise=0.15),
                            n=n, opening_plies=4)
    print(f"{'Lv.0 学習前(ランダム)':<20} | {w_rand:9.1f}% | {w_rb:11.1f}% | "
          f"{w_open:13.1f}% | {len(seen0):8}")

    for label, path in LEVELS:
        agent = DQNAgent()
        agent.load(path)
        w_rand, _ = eval_vs(agent, env, RandomAgent(), n=n)
        w_rb, _ = eval_vs(agent, env, StochasticRuleBasedAgent(noise=0.15), n=n)
        w_open, seen = eval_vs(agent, env, StochasticRuleBasedAgent(noise=0.15),
                               n=n, opening_plies=4)
        print(f"{label:<20} | {w_rand:9.1f}% | {w_rb:11.1f}% | "
              f"{w_open:13.1f}% | {len(seen):8}")


class _Wrap:
    """eval_vs は agent.epsilon を触るため、RandomAgent に属性を持たせる薄いラッパ。"""

    def __init__(self, inner):
        self._inner = inner
        self.epsilon = 0.0

    def __getattr__(self, name):
        return getattr(self._inner, name)


if __name__ == "__main__":
    main()
