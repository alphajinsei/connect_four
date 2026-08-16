"""
export_onnx.py — 学習済みDQNの重みを ONNX 形式に変換する（Web公開用）

なぜ ONNX が必要か:
    Cloudflare Pages は静的ファイルしか置けず、Python/PyTorch が動かない。
    ONNX は「NNの構造と重みを表現する共通ファイル形式」で、
    ブラウザ側の ONNX Runtime Web (JS) が読んで推論を実行できる。
    → サーバー不要でブラウザ内推論が可能になる。

重要:
    変換しただけでは不十分。数値誤差で PyTorch と違う手を選ぶ可能性があるため、
    **必ず同一性を検証する**（このスクリプトは変換後に自動で検証まで行う）。

使い方:
    .venv/Scripts/python 3moku/export_onnx.py
    → web-onnx/model.onnx が生成される
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.dqn_agent import QNetwork
from env.connect3_env import Connect3Env
from agents.rule_based_agent import RuleBasedAgent

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
WEIGHTS_PATH = os.path.join(_SCRIPT_DIR, "weights", "dqn_connect3.pt")
OUT_DIR = os.path.join(_REPO_ROOT, "web-onnx")
OUT_PATH = os.path.join(OUT_DIR, "model.onnx")

N_VERIFY = 2000  # 検証に使う局面数


def export():
    os.makedirs(OUT_DIR, exist_ok=True)

    net = QNetwork(rows=5, cols=5, output_size=5)
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
    net.eval()

    dummy = torch.zeros(1, 3, 5, 5, dtype=torch.float32)
    torch.onnx.export(
        net,
        dummy,
        OUT_PATH,
        input_names=["board"],
        output_names=["q_values"],
        # バッチ次元を可変にしておく（将来まとめて推論したくなった場合に備える）
        dynamic_axes={"board": {0: "batch"}, "q_values": {0: "batch"}},
        opset_version=17,
    )

    # torch 2.10 は重みを model.onnx.data に分離して出力する。
    # Web配信で2ファイル構成は事故のもと（片方だけキャッシュされる等）なので1つに統合する。
    import onnx
    from onnx.external_data_helper import load_external_data_for_model

    model = onnx.load(OUT_PATH, load_external_data=False)
    load_external_data_for_model(model, OUT_DIR)
    onnx.save(model, OUT_PATH, save_as_external_data=False)

    data_file = OUT_PATH + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"変換完了: {OUT_PATH} ({size_kb:.0f} KB, 単一ファイル)")
    return net


def collect_states(n):
    """実際の対局で現れる局面を集める（一様ランダムな盤面では現実の分布とズレるため）。"""
    env = Connect3Env()
    rb = RuleBasedAgent()
    states = []
    rng = np.random.default_rng(0)
    while len(states) < n:
        env.reset()
        # 開幕をランダム化して多様な局面を集める
        for _ in range(int(rng.integers(0, 5))):
            if env.done:
                break
            env.step(int(rng.choice(env.get_valid_actions())))
        while not env.done and len(states) < n:
            st = env.get_state(perspective=env.current_player)
            states.append(st.copy())
            va = env.get_valid_actions()
            if env.current_player == Connect3Env.PLAYER1:
                a = int(rng.choice(va))
            else:
                a = rb.get_action(st, va)
            env.step(a)
    return np.array(states[:n], dtype=np.float32)


def verify(net):
    """PyTorch と ONNX Runtime が同じ手を選ぶことを確認する。"""
    import onnxruntime as ort

    sess = ort.InferenceSession(OUT_PATH, providers=["CPUExecutionProvider"])
    states = collect_states(N_VERIFY)

    with torch.no_grad():
        q_torch = net(torch.from_numpy(states)).numpy()
    q_onnx = sess.run(["q_values"], {"board": states})[0]

    max_diff = float(np.abs(q_torch - q_onnx).max())

    # Q値そのものより「選ぶ手が同じか」が本質。
    # 無効手マスキングも含めて、実際の get_action と同じ手順で比較する。
    def choose(q_batch, boards):
        picks = []
        for q, b in zip(q_batch, boards):
            # b[0]は自分のコマ, b[1]は相手のコマ。合算して埋まっている列を判定
            occupied_top = (b[0][0] + b[1][0]) > 0
            valid = [c for c in range(5) if not occupied_top[c]]
            masked = np.full(5, -np.inf, dtype=np.float32)
            masked[valid] = q[valid]
            picks.append(int(np.argmax(masked)))
        return np.array(picks)

    pick_torch = choose(q_torch, states)
    pick_onnx = choose(q_onnx, states)
    agree = int((pick_torch == pick_onnx).sum())

    print()
    print(f"検証（{N_VERIFY}局面）:")
    print(f"  Q値の最大差   : {max_diff:.3e}")
    print(f"  選んだ手の一致 : {agree}/{N_VERIFY} ({agree / N_VERIFY * 100:.2f}%)")

    if agree != N_VERIFY:
        print("  → 不一致あり。ブラウザ版の挙動がPyTorch版と変わる可能性がある", file=sys.stderr)
        return False
    print("  → 完全一致。ブラウザ版はPyTorch版と同じ手を打つ")
    return True


if __name__ == "__main__":
    net = export()
    ok = verify(net)
    sys.exit(0 if ok else 1)
