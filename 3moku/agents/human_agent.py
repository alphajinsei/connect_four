from agents.base_agent import BaseAgent


class HumanAgent(BaseAgent):
    """
    is_human フラグを持つエージェント。
    実際の入力処理は GameRunner が CLIRenderer 経由で行う。
    """
    is_human = True

    def get_action(self, state, valid_actions):
        raise NotImplementedError("Human input is handled by the renderer in GameRunner.")
