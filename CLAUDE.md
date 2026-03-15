# Connect Four 強化学習プロジェクト

## ユーザーの目標
強化学習（DQN）を勉強中。Connect Four（四目並べ）を題材に、AIが学習して強くなる過程を実感したい。

## 学習ロードマップ
段階的に以下の順で進める：
1. **相手を固定**（ランダムエージェント） ← 現在ここ
2. **カリキュラム学習**（段階的に強い相手と対戦）
3. **Self-play**（自分自身と対戦して強化）

## 実行環境
- Python仮想環境: `.venv/`（プロジェクトルートに配置）
- **Pythonの実行は必ず `.venv` を使うこと**
  - 実行: `.venv/Scripts/python`
  - pip: `.venv/Scripts/pip`
- インストール済みパッケージ: `numpy`, `flask`

## ファイル構成

```
4moku/
├── env/
│   └── connect4_env.py     # ゲームロジック（重力・勝利判定・状態表現）
├── agents/
│   ├── base_agent.py       # 抽象基底クラス
│   ├── human_agent.py      # 人間プレイヤー（WebUI経由）
│   ├── random_agent.py     # ランダムAI（ステージ1の対戦相手）
│   └── dqn_agent.py        # ← 未実装（次のステップ）
├── ui/
│   └── cli_ui.py           # CLI表示（現在はWebUIがメイン）
├── web/
│   ├── app.py              # Flask サーバー
│   └── templates/
│       └── index.html      # ブラウザUI（アニメーション付き）
├── game_runner.py          # ターン管理（env・agent・UIを繋ぐ）
├── play.py                 # CLIで人間 vs ランダムAI
└── CLAUDE.md               # このファイル
```

## Webアプリの起動方法
```bash
cd c:\Users\peinn\OneDrive\sandbox\claude-sandbox\RL_ReinforcementLearning\4moku
.venv\Scripts\python web/app.py
# ブラウザで http://localhost:5000 を開く
```

## 設計の重要ポイント

### env/connect4_env.py
- `step()` は相手を呼び出さない（self-playやカリキュラム学習に対応するため）
- `get_state(perspective)` で視点を切り替えられる（DQN訓練・self-playで重要）
- 報酬は常に PLAYER1 視点で返し、`GameRunner` が各エージェント視点に変換する
- 状態は shape `(3, 6, 7)` の float32 配列
  - ch0: 自分のコマ
  - ch1: 相手のコマ
  - ch2: 自分のターンなら 1.0、そうでなければ 0.0

### game_runner.py
- agent1 = PLAYER1 (X / 赤)、agent2 = PLAYER2 (O / 橙)
- `GameRunner(env, dqn_agent, random_agent)` でステージ1
- `GameRunner(env, dqn_agent, dqn_agent)` でself-play（同一インスタンスも可）

## 次のステップ：dqn_agent.py の実装
- 入力サイズ: `3 × 6 × 7 = 126`
- 出力サイズ: `7`（列インデックス）
- アーキテクチャ: `Linear(126→256) → ReLU → Linear(256→256) → ReLU → Linear(256→7)`
- NumPyベース（PyTorch不使用）で実装する予定
- ε-greedy + 無効手マスキング
- ReplayBuffer、ターゲットネットワーク付き
- 重みの保存・読み込み（`np.savez` / `np.load`）
- 学習後は `web/app.py` の対戦相手を `RandomAgent` → `DQNAgent` に差し替える
