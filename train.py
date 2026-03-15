"""
train.py — DQN エージェントの学習スクリプト

使い方:
    python train.py
    python train.py --episodes 10000 --eval-interval 500
"""
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from env.connect4_env import Connect4Env
from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from game_runner import GameRunner

WEIGHTS_PATH = "weights/dqn_connect4"


def train(num_episodes=10000, eval_interval=500):
    env     = Connect4Env()
    agent   = DQNAgent(
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,   # 速める: ~100kステップで ε≈0.05 に到達
        buffer_capacity=50000,  # バッファを大きくして多様な経験を保持
        batch_size=128,         # バッチを大きくして安定した勾配
        warmup_steps=2000,      # バッファを十分蓄積してから学習開始
        target_update_interval=200,  # ターゲットネットをより頻繁に更新
    )
    opponent = RandomAgent()
    runner   = GameRunner(env, agent, opponent, renderer=None)

    win_history    = []
    reward_history = []

    print("学習開始（相手: ランダムAI固定）")
    print(f"{'Episode':>8} | {'勝率(直近500)':>14} | {'平均報酬':>10} | {'ε':>7}")
    print("-" * 52)

    for episode in range(1, num_episodes + 1):
        stats = runner.run_episode()
        win_history.append(1 if stats["winner"] == Connect4Env.PLAYER1 else 0)
        reward_history.append(stats["reward_p1"])

        if episode % eval_interval == 0:
            recent = win_history[-500:]
            win_rate = np.mean(recent) * 100
            avg_reward = np.mean(reward_history[-500:])
            print(f"{episode:>8} | {win_rate:>13.1f}% | {avg_reward:>10.3f} | {agent.epsilon:>7.5f}")

    print("\n学習完了")
    agent.save(WEIGHTS_PATH)
    print(f"重みを保存しました: {WEIGHTS_PATH}.npz")

    # テキストで勝率推移を表示
    print("\n=== 勝率推移（直近500エピソード平均）===")
    window = 500
    for i in range(window, len(win_history) + 1, window):
        wr = np.mean(win_history[i-window:i]) * 100
        bar = '#' * int(wr / 2)
        print(f"Ep {i:>6}: {bar:<50} {wr:.1f}%")

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",      type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=500)
    args = parser.parse_args()

    train(num_episodes=args.episodes, eval_interval=args.eval_interval)
