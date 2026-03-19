"""
web/app.py — AlphaZero Connect Four Web UI (Flask)

起動方法:
    .venv/Scripts/python alphazero/web/app.py
    ブラウザで http://localhost:5001 を開く

4moku版との違い:
  - DQN: state→Q値で即座に手を決定（<1ms）
  - AlphaZero: MCTS(50シミュレーション)で手を決定（〜1秒）
    → UIに「思考中...」表示が必要
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, render_template, jsonify, request, session
from env.connect4_env import Connect4Env
from agents.alphazero_agent import AlphaZeroAgent

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(_APP_DIR, "weights", "alphazero_latest.pt")

NUM_SIMULATIONS = 50


def make_opponent():
    if os.path.exists(WEIGHTS_PATH):
        agent = AlphaZeroAgent(weights_path=WEIGHTS_PATH, num_simulations=NUM_SIMULATIONS)
        return agent
    print("[WebUI] weights が見つかりません。ランダムAIを使用します。")
    return None


app = Flask(__name__)
app.secret_key = "alphazero-c4-secret"

games = {}
ai_agent = None


def get_or_create_agent():
    global ai_agent
    if ai_agent is None:
        ai_agent = make_opponent()
    return ai_agent


def get_game(session_id):
    if session_id not in games:
        env = Connect4Env()
        env.reset()
        games[session_id] = {"env": env}
    return games[session_id]


def env_to_dict(env):
    return {
        "board": env.board.tolist(),
        "current_player": int(env.current_player),
        "done": env.done,
        "winner": int(env.winner) if env.winner is not None else None,
        "valid_actions": env.get_valid_actions(),
    }


@app.route("/")
def index():
    session.setdefault("id", os.urandom(8).hex())
    return render_template("index.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    sid = session.get("id", "default")
    env = Connect4Env()
    env.reset()
    games[sid] = {"env": env}

    agent = get_or_create_agent()

    # AIが先攻 (PLAYER1)
    ai_col = None
    ai_row = None
    if agent is not None:
        ai_col = agent.get_action(env)
        ai_row = env._apply_gravity(ai_col)
        env.step(ai_col)
    else:
        # ランダム
        import numpy as np
        valid = env.get_valid_actions()
        ai_col = int(np.random.choice(valid))
        ai_row = env._apply_gravity(ai_col)
        env.step(ai_col)

    result = env_to_dict(env)
    result["ai_col"] = ai_col
    result["ai_row"] = ai_row
    return jsonify(result)


@app.route("/api/move", methods=["POST"])
def move():
    sid = session.get("id", "default")
    game = get_game(sid)
    env = game["env"]
    agent = get_or_create_agent()

    if env.done:
        return jsonify({"error": "game is already over"}), 400

    data = request.get_json()
    col = data.get("col")
    if col is None or col not in env.get_valid_actions():
        return jsonify({"error": "invalid move"}), 400

    # 人間の手（PLAYER2）
    human_row = env._apply_gravity(col)
    done, winner = env.step(col)

    # AIの手（ゲームが終わっていなければ）
    ai_col = None
    ai_row = None
    if not done:
        if agent is not None:
            ai_col = agent.get_action(env)
        else:
            import numpy as np
            valid = env.get_valid_actions()
            ai_col = int(np.random.choice(valid))
        ai_row = env._apply_gravity(ai_col)
        env.step(ai_col)

    result = env_to_dict(env)
    result["human_row"] = human_row
    result["ai_col"] = ai_col
    result["ai_row"] = ai_row
    return jsonify(result)


if __name__ == "__main__":
    get_or_create_agent()
    app.run(debug=False, port=5001)
