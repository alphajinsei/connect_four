"""
Connect Four ゲーム環境（AlphaZero用）

4moku版をベースに、AlphaZero向けにシンプル化:
- 中間報酬なし（AlphaZeroは勝敗のみで学習）
- get_canonical_state() で「現在のプレイヤー視点」の盤面を返す
- clone() でMCTSシミュレーション用のコピーを作成可能
"""
import numpy as np


class Connect4Env:
    ROWS = 6
    COLS = 7
    WIN_LENGTH = 4
    NUM_ACTIONS = 7

    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = -1

    def __init__(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = self.PLAYER1
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = self.PLAYER1
        self.done = False
        self.winner = None
        return self

    def clone(self):
        """MCTS用: 現在の状態のコピーを返す"""
        env = Connect4Env()
        env.board = self.board.copy()
        env.current_player = self.current_player
        env.done = self.done
        env.winner = self.winner
        return env

    def get_valid_actions(self):
        return [c for c in range(self.COLS) if self.board[0][c] == self.EMPTY]

    def get_valid_actions_mask(self):
        """合法手マスク (NUM_ACTIONS,) を返す。合法=1, 違法=0"""
        return np.array([1 if self.board[0][c] == self.EMPTY else 0
                         for c in range(self.COLS)], dtype=np.float32)

    def step(self, col):
        """
        手を打ち、(done, winner) を返す。
        winner: PLAYER1, PLAYER2, 0(引き分け), None(継続中)
        """
        if self.done:
            raise RuntimeError("Episode is already done. Call reset().")
        if self.board[0][col] != self.EMPTY:
            raise ValueError(f"Column {col} is full.")

        row = self._apply_gravity(col)
        self.board[row][col] = self.current_player

        if self._check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            return True, self.winner

        if self._is_full():
            self.done = True
            self.winner = 0
            return True, 0

        self.current_player = -self.current_player
        return False, None

    def get_canonical_state(self):
        """
        現在のプレイヤー視点の状態を返す。shape: (3, ROWS, COLS)
          ch0: 現在のプレイヤーのコマ (1.0)
          ch1: 相手のプレイヤーのコマ (1.0)
          ch2: 全て1.0（常に「自分のターン」）

        AlphaZeroでは常に「自分のターン」の状態をNNに渡すため、
        ch2は常に1.0で固定。
        """
        p = self.current_player
        state = np.zeros((3, self.ROWS, self.COLS), dtype=np.float32)
        state[0] = (self.board == p).astype(np.float32)
        state[1] = (self.board == -p).astype(np.float32)
        state[2] = 1.0
        return state

    def _apply_gravity(self, col):
        for row in range(self.ROWS - 1, -1, -1):
            if self.board[row][col] == self.EMPTY:
                return row
        raise ValueError(f"Column {col} is full.")

    def _check_win(self, player):
        b = self.board
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                if all(b[r][c + i] == player for i in range(4)):
                    return True
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                if all(b[r + i][c] == player for i in range(4)):
                    return True
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                if all(b[r + i][c + i] == player for i in range(4)):
                    return True
        for r in range(self.ROWS - 3):
            for c in range(3, self.COLS):
                if all(b[r + i][c - i] == player for i in range(4)):
                    return True
        return False

    def _is_full(self):
        return len(self.get_valid_actions()) == 0

    def render_board(self):
        col_nums = " ".join(str(c) for c in range(self.COLS))
        lines = [f" {col_nums}"]
        for r in range(self.ROWS):
            row_str = "|"
            for c in range(self.COLS):
                v = self.board[r][c]
                if v == self.PLAYER1:
                    row_str += "X"
                elif v == self.PLAYER2:
                    row_str += "O"
                else:
                    row_str += "."
                row_str += "|" if c == self.COLS - 1 else " "
            lines.append(row_str)
        return "\n".join(lines)
