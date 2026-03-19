import random
import numpy as np
from agents.base_agent import BaseAgent


class RuleBasedAgent(BaseAgent):
    """
    ルールベースAI（Connect Three用）。学習なし、手書きルールで動く。

    優先順位:
      1. 自分が即勝てる列があれば打つ
      2. 相手が次に勝てる列があれば防ぐ
      3. フォーク（1手で2箇所以上リーチ）を狙う
      4. 相手のフォークを防ぐ
      5. 打った上に相手が置くと相手が勝つ列を避ける（トラップ回避）
      6. リーチ（2連で両端が空き）を作る手を優先
      7. 中央寄りの列を優先
    """

    WIN_LENGTH = 3

    def get_action(self, state, valid_actions):
        my_board = state[0]
        opp_board = state[1]
        rows, cols = my_board.shape

        board = np.zeros((rows, cols), dtype=np.int8)
        board[my_board == 1] = 1
        board[opp_board == 1] = -1

        # 1. 自分の勝ち手
        for col in valid_actions:
            if self._wins_if_placed(board, rows, cols, col, 1):
                return col

        # 2. 相手の勝ち手を防ぐ
        for col in valid_actions:
            if self._wins_if_placed(board, rows, cols, col, -1):
                return col

        # 3. フォークを狙う（1手で2箇所以上リーチを作る）
        for col in valid_actions:
            if self._creates_fork(board, rows, cols, col, 1):
                return col

        # 4. 相手のフォークを防ぐ
        for col in valid_actions:
            if self._creates_fork(board, rows, cols, col, -1):
                return col

        # 5. トラップ回避: 打った上に相手が置くと相手が勝つ列を除外
        safe_actions = []
        for col in valid_actions:
            row = self._top_row(board, rows, col)
            if row is not None and row > 0:
                # 自分が打った上のマスに相手が置いたら勝たれるか？
                board[row][col] = 1
                if self._wins_if_placed(board, rows, cols, col, -1):
                    board[row][col] = 0
                    continue
                board[row][col] = 0
            safe_actions.append(col)

        # 安全な手がなければ全候補を使う
        candidates = safe_actions if safe_actions else valid_actions

        # 6. リーチ（開放2連）を作る手にスコアリング
        best_score = -1
        best_cols = []
        center = cols // 2
        for col in candidates:
            score = self._threat_score(board, rows, cols, col, 1)
            # 中央寄りに微小ボーナス
            score += (cols - abs(col - center)) * 0.1
            if score > best_score:
                best_score = score
                best_cols = [col]
            elif score == best_score:
                best_cols.append(col)

        return random.choice(best_cols)

    def _top_row(self, board, rows, col):
        for row in range(rows - 1, -1, -1):
            if board[row][col] == 0:
                return row
        return None

    def _wins_if_placed(self, board, rows, cols, col, player):
        row = self._top_row(board, rows, col)
        if row is None:
            return False
        board[row][col] = player
        result = self._check_win(board, rows, cols, player)
        board[row][col] = 0
        return result

    def _count_threats(self, board, rows, cols, player):
        """playerが次の1手で勝てる列の数を数える。"""
        threats = 0
        for c in range(cols):
            if self._wins_if_placed(board, rows, cols, c, player):
                threats += 1
        return threats

    def _creates_fork(self, board, rows, cols, col, player):
        """colに打つと、playerが2箇所以上で即勝ちできる状態になるか。"""
        row = self._top_row(board, rows, col)
        if row is None:
            return False
        board[row][col] = player
        threats = self._count_threats(board, rows, cols, player)
        board[row][col] = 0
        return threats >= 2

    def _threat_score(self, board, rows, cols, col, player):
        """colに打った後、playerのリーチ数を返す。"""
        row = self._top_row(board, rows, col)
        if row is None:
            return 0
        board[row][col] = player
        score = self._count_threats(board, rows, cols, player)
        board[row][col] = 0
        return score

    def _check_win(self, board, rows, cols, player):
        wl = self.WIN_LENGTH
        for r in range(rows):
            for c in range(cols - wl + 1):
                if all(board[r][c + i] == player for i in range(wl)):
                    return True
        for r in range(rows - wl + 1):
            for c in range(cols):
                if all(board[r + i][c] == player for i in range(wl)):
                    return True
        for r in range(rows - wl + 1):
            for c in range(cols - wl + 1):
                if all(board[r + i][c + i] == player for i in range(wl)):
                    return True
        for r in range(rows - wl + 1):
            for c in range(wl - 1, cols):
                if all(board[r + i][c - i] == player for i in range(wl)):
                    return True
        return False
