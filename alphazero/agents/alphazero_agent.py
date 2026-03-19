"""
AlphaZeroAgent — WebUI・対戦用エージェント

MCTSを使って手を選択する。学習は行わない（推論専用）。
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from network import AlphaZeroNet
from mcts import MCTS
from env.connect4_env import Connect4Env


class AlphaZeroAgent:
    """
    AlphaZero推論エージェント。

    DQNAgentとの違い:
      - DQN: state → Q値 → argmax で即座に手を決定（推論は一瞬）
      - AlphaZero: state → MCTS（数十回のシミュレーション）→ 訪問回数で手を決定
        → 推論は遅いが、先読みがある分はるかに強い
    """

    def __init__(self, weights_path=None, num_simulations=50):
        self.network = AlphaZeroNet(num_res_blocks=4, channels=64)
        if weights_path and os.path.exists(weights_path):
            self.network.load_state_dict(
                torch.load(weights_path, map_location='cpu', weights_only=True)
            )
            print(f"[AlphaZeroAgent] 重みをロード: {weights_path}")
        self.network.eval()
        self.mcts = MCTS(self.network, num_simulations=num_simulations, c_puct=1.5)

    def get_action(self, env):
        """
        現在のenv状態からMCTSで最善手を返す。

        Args:
            env: Connect4Env（変更されない）
        Returns:
            action: int (列番号)
        """
        action_probs = self.mcts.search(env, add_noise=False)
        return int(np.argmax(action_probs))
