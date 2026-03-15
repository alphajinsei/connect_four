import numpy as np


class Connect4Env:
    ROWS = 6
    COLS = 7
    WIN_LENGTH = 4

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
        return self.get_state()

    def get_valid_actions(self):
        return [c for c in range(self.COLS) if self.board[0][c] == self.EMPTY]

    def step(self, col):
        """
        Returns (state, reward, done, info).
        reward は常に PLAYER1 視点。
        """
        if self.done:
            raise RuntimeError("Episode is already done. Call reset().")

        if col not in self.get_valid_actions():
            self.done = True
            reward = -1.0 if self.current_player == self.PLAYER1 else 1.0
            return self.get_state(), reward, True, {"winner": None, "invalid_move": True}

        row = self._apply_gravity(col)
        self.board[row][col] = self.current_player

        if self._check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1.0 if self.current_player == self.PLAYER1 else -1.0
            return self.get_state(), reward, True, {"winner": self.winner, "invalid_move": False}

        if self._is_full():
            self.done = True
            self.winner = 0  # draw
            return self.get_state(), 0.0, True, {"winner": 0, "invalid_move": False}

        self.current_player = self.PLAYER2 if self.current_player == self.PLAYER1 else self.PLAYER1
        return self.get_state(), 0.0, False, {"winner": None, "invalid_move": False}

    def get_state(self, perspective=None):
        """
        Shape (3, ROWS, COLS) の状態を返す。
          ch0: perspective のコマ
          ch1: 相手のコマ
          ch2: perspective のターンなら 1.0、そうでなければ 0.0
        """
        if perspective is None:
            perspective = self.current_player
        opponent = self.PLAYER2 if perspective == self.PLAYER1 else self.PLAYER1

        state = np.zeros((3, self.ROWS, self.COLS), dtype=np.float32)
        state[0] = (self.board == perspective).astype(np.float32)
        state[1] = (self.board == opponent).astype(np.float32)
        state[2] = 1.0 if self.current_player == perspective else 0.0
        return state

    def _apply_gravity(self, col):
        for row in range(self.ROWS - 1, -1, -1):
            if self.board[row][col] == self.EMPTY:
                return row
        raise ValueError(f"Column {col} is full.")

    def _check_win(self, player):
        b = self.board
        # 横
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                if all(b[r][c + i] == player for i in range(4)):
                    return True
        # 縦
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                if all(b[r + i][c] == player for i in range(4)):
                    return True
        # 斜め右下
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                if all(b[r + i][c + i] == player for i in range(4)):
                    return True
        # 斜め左下
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
