"""
web/app.py — Connect Three Web UI (Flask)

起動方法:
    python web/app.py
    ブラウザで http://localhost:5001 を開く
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, render_template, jsonify, request, session
from env.connect3_env import Connect3Env
from agents.random_agent import RandomAgent
from agents.dqn_agent import DQNAgent

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(_APP_DIR, "weights", "dqn_connect3.pt")


def make_opponent():
    if os.path.exists(WEIGHTS_PATH):
        agent = DQNAgent(epsilon_start=0.0, epsilon_end=0.0)
        agent.load(WEIGHTS_PATH)
        print(f"[WebUI] DQNAgent を読み込みました: {WEIGHTS_PATH}")
        return agent
    print("[WebUI] weights が見つかりません。RandomAgent を使用します。")
    return RandomAgent()

app = Flask(__name__)
app.secret_key = "connect3-secret"

games = {}


def get_game(session_id):
    if session_id not in games:
        env = Connect3Env()
        env.reset()
        games[session_id] = {
            "env": env,
            "opponent": make_opponent(),
            "player_is": Connect3Env.PLAYER2,
        }
    return games[session_id]


def env_to_dict(env):
    board = env.board.tolist()
    return {
        "board": board,
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
    if sid in games:
        games[sid]["env"].reset()
    else:
        get_game(sid)
    game = games[sid]
    env = game["env"]
    env.reset()

    opponent = game["opponent"]
    state = env.get_state(perspective=env.current_player)
    valid = env.get_valid_actions()
    ai_col = opponent.get_action(state, valid)
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
    opponent = game["opponent"]

    if env.done:
        return jsonify({"error": "game is already over"}), 400

    data = request.get_json()
    col = data.get("col")
    if col is None or col not in env.get_valid_actions():
        return jsonify({"error": "invalid move"}), 400

    human_row = env._apply_gravity(col)
    _, reward, done, info = env.step(col)

    ai_col = None
    ai_row = None
    if not done:
        state = env.get_state(perspective=env.current_player)
        valid = env.get_valid_actions()
        ai_col = opponent.get_action(state, valid)
        ai_row = env._apply_gravity(ai_col)
        _, reward, done, info = env.step(ai_col)

    result = env_to_dict(env)
    result["human_row"] = human_row
    result["ai_col"] = ai_col
    result["ai_row"] = ai_row
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
