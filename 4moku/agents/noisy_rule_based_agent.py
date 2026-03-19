import random
import numpy as np
from agents.rule_based_agent import RuleBasedAgent


class NoisyRuleBasedAgent(RuleBasedAgent):
    """
    ルールベースAIに noise を加えたエージェント。

    noise=0.0: 完全なルールベースAI
    noise=1.0: 完全ランダム
    noise=0.3: 30% の確率でランダム手を選ぶ（残り70%はルールに従う）
    """

    def __init__(self, noise=0.5):
        self.noise = noise

    def get_action(self, state, valid_actions):
        if random.random() < self.noise:
            return random.choice(valid_actions)
        return super().get_action(state, valid_actions)
