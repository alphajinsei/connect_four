import random
import numpy as np
from agents.base_agent import BaseAgent


class RuleBasedAgent(BaseAgent):
    """
    ルールベースAI。学習なし、手書きルールで動く。

    優先順位:
      1. 自分が即勝てる列があれば打つ
      2. 相手が次に勝てる列があれば防ぐ
      3. 中央寄りの列を優先してランダム選択
    """

    def get_action(self, state, valid_actions):
        # state は (3, ROWS, COLS) だが、盤面情報は env.board を直接使いたい。
        # ここでは state から盤面を復元する。
        # ch0=自分のコマ、ch1=相手のコマ
        my_board    = state[0]   # shape (ROWS, COLS)
        opp_board   = state[1]
        rows, cols  = my_board.shape

        board = np.zeros((rows, cols), dtype=np.int8)
        board[my_board  == 1] = 1   # 自分 = 1
        board[opp_board == 1] = -1  # 相手 = -1

        # ---- ヘルパー: ある列に player を置いたとき勝てるか ----
        def wins_if_placed(col, player):
            row = self._top_row(board, rows, col)
            if row is None:
                return False
            board[row][col] = player
            result = self._check_win(board, rows, cols, player)
            board[row][col] = 0
            return result

        # 1. 自分の勝ち手
        for col in valid_actions:
            if wins_if_placed(col, 1):
                return col

        # 2. 相手の勝ち手を防ぐ
        for col in valid_actions:
            if wins_if_placed(col, -1):
                return col

        # 3. 中央寄り優先（中央から順に優先度を付けてシャッフル）
        center = cols // 2
        weighted = sorted(valid_actions, key=lambda c: abs(c - center))
        # 同優先度内はランダム性を持たせる
        groups = {}
        for c in weighted:
            d = abs(c - center)
            groups.setdefault(d, []).append(c)
        ordered = []
        for d in sorted(groups):
            g = groups[d]
            random.shuffle(g)
            ordered.extend(g)
        return ordered[0]

    # ---- 内部ユーティリティ ----

    def _top_row(self, board, rows, col):
        """指定列に置いたときの行インデックス（重力）。満杯なら None。"""
        for row in range(rows - 1, -1, -1):
            if board[row][col] == 0:
                return row
        return None

    def _check_win(self, board, rows, cols, player):
        """board 上で player が4連しているか判定。"""
        # 横
        for r in range(rows):
            for c in range(cols - 3):
                if all(board[r][c + i] == player for i in range(4)):
                    return True
        # 縦
        for r in range(rows - 3):
            for c in range(cols):
                if all(board[r + i][c] == player for i in range(4)):
                    return True
        # 斜め右下
        for r in range(rows - 3):
            for c in range(cols - 3):
                if all(board[r + i][c + i] == player for i in range(4)):
                    return True
        # 斜め左下
        for r in range(rows - 3):
            for c in range(3, cols):
                if all(board[r + i][c - i] == player for i in range(4)):
                    return True
        return False
