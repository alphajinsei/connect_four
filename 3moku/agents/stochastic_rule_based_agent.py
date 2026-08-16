import random

from agents.rule_based_agent import RuleBasedAgent


class StochasticRuleBasedAgent(RuleBasedAgent):
    """
    RuleBasedAgent に確率的な揺らぎを加えたもの。**弱くするためではなく、
    対局の分岐を増やして DQN の訪問局面を広げるのが目的。**

    背景: 決定論的な RuleBasedAgent と決定論的な DQN(ε=0) が対戦すると、
    300ゲームで棋譜が2通り・DQN が遭遇する盤面は11種類しかなかった。
    vs RuleBased 100% は「11局面の暗記テスト満点」に過ぎず、人間の変則手に
    対応できない原因になっていた。

    4moku の NoisyRuleBasedAgent とは意図が異なる点に注意:
      - NoisyRuleBased: カリキュラム用。noise を段階的に下げていく → 失敗
      - この実装: 分岐生成用。noise は学習を通じて固定

    noise 手を打つ場合も「自分の即勝ち」「相手の即勝ちの阻止」だけは維持する。
    そこまで崩すと相手が弱くなりすぎ、DQN が防御を学ぶ動機を失うため
    （3moku Stage 2 で確認済みの失敗パターン）。
    """

    def __init__(self, noise=0.15):
        self.noise = noise

    def get_action(self, state, valid_actions):
        if random.random() >= self.noise:
            return super().get_action(state, valid_actions)

        # noise 手: 勝ち/負けに直結する手だけは崩さない
        import numpy as np
        my_board = state[0]
        opp_board = state[1]
        rows, cols = my_board.shape
        board = np.zeros((rows, cols), dtype=np.int8)
        board[my_board == 1] = 1
        board[opp_board == 1] = -1

        for col in valid_actions:
            if self._wins_if_placed(board, rows, cols, col, 1):
                return col
        for col in valid_actions:
            if self._wins_if_placed(board, rows, cols, col, -1):
                return col

        return random.choice(valid_actions)
