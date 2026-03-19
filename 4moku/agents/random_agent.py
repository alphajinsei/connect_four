import random
from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def get_action(self, state, valid_actions):
        return random.choice(valid_actions)
