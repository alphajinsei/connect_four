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

        # 中間報酬なし（勝敗報酬±1.0のみ）
        # CNN導入により空間パターン認識が可能になったため、中間報酬は不要
        shaping = 0.0
        self.current_player = self.PLAYER2 if self.current_player == self.PLAYER1 else self.PLAYER1
        return self.get_state(), shaping, False, {"winner": None, "invalid_move": False}

    def _shaping_reward(self, player, row, col):
        """
        置いた (row, col) を起点に攻防セットの中間報酬を返す（PLAYER1 視点）。

        報酬設計（極小スケール — 1ゲーム累積 ≪ 勝敗±1.0）:
          自分の3連を作った    : +0.03
          相手の3連を防いだ    : +0.02
          相手の勝ち手を見逃した: -0.05
          2連の報酬は廃止（ノイズになるため）
        """
        b = self.board
        opponent = self.PLAYER2 if player == self.PLAYER1 else self.PLAYER1
        sign = 1 if player == self.PLAYER1 else -1

        def count_line(r0, c0, dr, dc, target):
            """(r0,c0) から両方向に連続する target のコマ数を返す（r0,c0 自身は含まない）"""
            n = 0
            for d in (1, -1):
                r, c = r0 + dr * d, c0 + dc * d
                while 0 <= r < self.ROWS and 0 <= c < self.COLS and b[r][c] == target:
                    n += 1
                    r += dr * d
                    c += dc * d
            return n

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # 自分の連（置いた直後の自分コマ起点）
        my_max = max(count_line(row, col, dr, dc, player) for dr, dc in directions)

        # 相手の連（置く前に相手が作っていた連を防いだか）
        # 置いたマスを一時的に空にして、相手視点の連を計る
        b[row][col] = self.EMPTY
        opp_max = max(count_line(row, col, dr, dc, opponent) for dr, dc in directions)
        b[row][col] = player  # 元に戻す

        # 攻撃報酬（3連のみ、2連は廃止）
        attack = 0.03 if my_max >= 3 else 0.0

        # 防御報酬（3連阻止のみ、2連は廃止）
        defense = 0.02 if opp_max >= 3 else 0.0

        # 防御失敗ペナルティ: 次のターンで相手が勝てる手があるか確認
        miss_penalty = 0.0
        for c in range(self.COLS):
            top = None
            for r in range(self.ROWS - 1, -1, -1):
                if b[r][c] == self.EMPTY:
                    top = r
                    break
            if top is None:
                continue
            b[top][c] = opponent
            if self._check_win(opponent):
                miss_penalty = -0.05
            b[top][c] = self.EMPTY
            if miss_penalty != 0.0:
                break

        shaping = attack + defense + miss_penalty
        return sign * shaping

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
