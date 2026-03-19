"""
play.py — 人間 vs ランダムAI で Connect Three を遊ぶ

使い方:
    python play.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from env.connect3_env import Connect3Env
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from ui.cli_ui import CLIRenderer
from game_runner import GameRunner


def main():
    print("=== Connect Three (三目並べ) ===")
    print("X = PLAYER1 (あなた)")
    print("O = PLAYER2 (ランダムAI)")
    print()

    env = Connect3Env()
    player1 = HumanAgent()
    player2 = RandomAgent()
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
