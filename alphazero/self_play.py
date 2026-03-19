"""
Self-Play データ生成

AlphaZeroのself-playの流れ:
  1. ゲーム開始
  2. 各手番でMCTSを実行 → 行動確率 π を得る
  3. 温度付きで行動を選択（序盤: temperature=1.0, 終盤: greedy）
  4. ゲーム終了後、各盤面に実際の勝敗結果 z を付与
  5. 学習データ: (state, π, z) のリスト

DQNとの根本的な違い:
  - DQN: 1手ごとに (s, a, r, s') を保存 → Q値を学習
  - AlphaZero: 1ゲーム全体を (s, π, z) として保存 → policyとvalueを同時に学習
"""
import numpy as np
from env.connect4_env import Connect4Env
from mcts import MCTS


def play_one_game(mcts, temperature_threshold=15):
    """
    1ゲームのself-playを実行し、学習データを返す。

    Args:
        mcts: MCTSインスタンス
        temperature_threshold: この手数まで temperature=1.0、以降は greedy

    Returns:
        game_data: [(state, action_probs, current_player), ...] のリスト
        winner: PLAYER1, PLAYER2, or 0 (draw)
    """
    env = Connect4Env()
    env.reset()

    game_data = []
    move_count = 0

    while not env.done:
        # MCTSで行動確率を計算
        action_probs = mcts.search(env, add_noise=True)

        # 状態を保存（現在のプレイヤー視点）
        state = env.get_canonical_state()
        game_data.append((state, action_probs, env.current_player))

        # 温度付きで行動を選択
        temperature = 1.0 if move_count < temperature_threshold else 0.0
        action = mcts.get_action_with_temperature(action_probs, temperature)

        env.step(action)
        move_count += 1

    return game_data, env.winner


def generate_training_data(game_data, winner):
    """
    self-playの結果を学習データに変換する。

    各 (state, action_probs, player) に対して:
      - z = +1.0 if player == winner
      - z = -1.0 if player != winner
      - z =  0.0 if draw

    Returns:
        states: (N, 3, H, W)
        policies: (N, num_actions)
        values: (N,)
    """
    states = []
    policies = []
    values = []

    for state, action_probs, player in game_data:
        states.append(state)
        policies.append(action_probs)

        if winner == 0:
            values.append(0.0)
        elif winner == player:
            values.append(1.0)
        else:
            values.append(-1.0)

    return (
        np.array(states, dtype=np.float32),
        np.array(policies, dtype=np.float32),
        np.array(values, dtype=np.float32),
    )


def self_play_batch(network, num_games, num_simulations=50, c_puct=1.5,
                    temperature_threshold=15):
    """
    複数ゲームのself-playを実行し、学習データを蓄積する。

    Returns:
        all_states: (total_moves, 3, H, W)
        all_policies: (total_moves, num_actions)
        all_values: (total_moves,)
        stats: dict with win/loss/draw counts
    """
    mcts = MCTS(network, num_simulations=num_simulations, c_puct=c_puct)

    all_states = []
    all_policies = []
    all_values = []
    stats = {"p1_wins": 0, "p2_wins": 0, "draws": 0}

    for _ in range(num_games):
        game_data, winner = play_one_game(mcts, temperature_threshold)
        states, policies, values = generate_training_data(game_data, winner)

        all_states.append(states)
        all_policies.append(policies)
        all_values.append(values)

        if winner == Connect4Env.PLAYER1:
            stats["p1_wins"] += 1
        elif winner == Connect4Env.PLAYER2:
            stats["p2_wins"] += 1
        else:
            stats["draws"] += 1

    return (
        np.concatenate(all_states),
        np.concatenate(all_policies),
        np.concatenate(all_values),
        stats,
    )
