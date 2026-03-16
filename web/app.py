"""
web/app.py — Connect Four Web UI (Flask)

起動方法:
    python web/app.py
    ブラウザで http://localhost:5000 を開く
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, render_template, jsonify, request, session
from env.connect4_env import Connect4Env
from agents.random_agent import RandomAgent
from agents.dqn_agent import DQNAgent

WEIGHTS_PATH = "weights/dqn_connect4.npz"


def make_opponent():
    """重みファイルがあれば DQNAgent、なければ RandomAgent を返す"""
    if os.path.exists(WEIGHTS_PATH):
        agent = DQNAgent(epsilon_start=0.0, epsilon_end=0.0)  # greedy
        agent.load(WEIGHTS_PATH)
        print(f"[WebUI] DQNAgent を読み込みました: {WEIGHTS_PATH}")
        return agent
    print("[WebUI] weights が見つかりません。RandomAgent を使用します。")
    return RandomAgent()

app = Flask(__name__)
app.secret_key = "connect4-secret"

# セッションごとにゲーム状態を保持するための簡易ストア
# （本格的なマルチユーザー対応はしない）
games = {}


def get_game(session_id):
    if session_id not in games:
        env = Connect4Env()
        env.reset()
        games[session_id] = {
            "env": env,
            "opponent": make_opponent(),
            "player_is": Connect4Env.PLAYER2,  # 人間は後攻 (PLAYER2)、AIが先攻 (PLAYER1)
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
        get_game(sid)  # 初期化
    game = games[sid]
    env = game["env"]
    env.reset()

    # AIが先攻 (PLAYER1) なので、ゲーム開始時に1手打つ
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

    # AIの手を先に打つ（AIは先攻 PLAYER1）
    # ※ /api/move が呼ばれる時点では人間(PLAYER2)のターンのはずだが、
    #   念のためターン確認はしない（UIが制御する）

    # 人間の手（PLAYER2）
    human_row = env._apply_gravity(col)
    _, reward, done, info = env.step(col)

    # AIの手（ゲームが終わっていなければ）
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
    app.run(debug=True, port=5000)
