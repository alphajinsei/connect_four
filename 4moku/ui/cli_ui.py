class CLIRenderer:
    def render(self, env):
        print()
        print(env.render_board())
        player_label = "X (PLAYER1)" if env.current_player == env.PLAYER1 else "O (PLAYER2)"
        if not env.done:
            print(f"次の手番: {player_label}")

    def prompt_human_action(self, valid_actions):
        while True:
            try:
                col = int(input(f"列を選んでください {valid_actions}: "))
                if col in valid_actions:
                    return col
                print(f"無効な列です。{valid_actions} の中から選んでください。")
            except ValueError:
                print("数字を入力してください。")

    def show_result(self, winner):
        print()
        if winner == 1:
            print("X (PLAYER1) の勝ち！")
        elif winner == -1:
            print("O (PLAYER2) の勝ち！")
        elif winner == 0:
            print("引き分け！")
        else:
            print("無効手により終了。")
