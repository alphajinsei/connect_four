import random
import numpy as np
from agents.base_agent import BaseAgent


class RuleBasedAgent(BaseAgent):
    """
    ルールベースAI（Connect Three用）。学習なし、手書きルールで動く。

    優先順位:
      1. 自分が即勝てる列があれば打つ
      2. 相手が次に勝てる列があれば防ぐ
      3. 中央寄りの列を優先してランダム選択
    """

    WIN_LENGTH = 3

    def get_action(self, state, valid_actions):
        my_board    = state[0]   # shape (ROWS, COLS)
        opp_board   = state[1]
        rows, cols  = my_board.shape

        board = np.zeros((rows, cols), dtype=np.int8)
        board[my_board  == 1] = 1   # 自分 = 1
        board[opp_board == 1] = -1  # 相手 = -1

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

        # 3. 中央寄り優先
        center = cols // 2
        weighted = sorted(valid_actions, key=lambda c: abs(c - center))
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

    def _top_row(self, board, rows, col):
        for row in range(rows - 1, -1, -1):
            if board[row][col] == 0:
                return row
        return None

    def _check_win(self, board, rows, cols, player):
        wl = self.WIN_LENGTH
        # 横
        for r in range(rows):
            for c in range(cols - wl + 1):
                if all(board[r][c + i] == player for i in range(wl)):
                    return True
        # 縦
        for r in range(rows - wl + 1):
            for c in range(cols):
                if all(board[r + i][c] == player for i in range(wl)):
                    return True
        # 斜め右下
        for r in range(rows - wl + 1):
            for c in range(cols - wl + 1):
                if all(board[r + i][c + i] == player for i in range(wl)):
                    return True
        # 斜め左下
        for r in range(rows - wl + 1):
            for c in range(wl - 1, cols):
                if all(board[r + i][c - i] == player for i in range(wl)):
                    return True
        return False
