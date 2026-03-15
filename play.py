"""
play.py — 人間対人間で Connect Four を遊ぶ

使い方:
    python play.py

後々 DQN エージェントが実装されたら、PLAYER2 を DQNAgent に差し替えるだけで
人間 vs AI になります。
"""
import sys
import os

# プロジェクトルートを sys.path に追加
sys.path.insert(0, os.path.dirname(__file__))

from env.connect4_env import Connect4Env
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from ui.cli_ui import CLIRenderer
from game_runner import GameRunner


def main():
    print("=== Connect Four (四目並べ) ===")
    print("X = PLAYER1 (あなた)")
    print("O = PLAYER2 (ランダムAI)")
    print()

    env = Connect4Env()
    player1 = HumanAgent()       # あなた (X)
    player2 = RandomAgent()      # ランダムAI (O)
    renderer = CLIRenderer()
    runner = GameRunner(env, player1, player2, renderer=renderer)

    while True:
        runner.run_episode()
        print()
        again = input("もう一度遊びますか？ (y/n): ").strip().lower()
        if again != "y":
            print("終了します。")
            break


if __name__ == "__main__":
    main()
