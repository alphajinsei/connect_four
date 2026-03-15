# Connect Four 強化学習プロジェクト

## ユーザーの目標
強化学習（DQN）を勉強中。Connect Four（四目並べ）を題材に、AIが学習して強くなる過程を実感したい。

## 学習ロードマップ
段階的に以下の順で進める：
1. **相手を固定**（ランダムエージェント） ← 完了（勝率 ~79% 達成）
2. **カリキュラム学習**（段階的に強い相手と対戦） ← 現在ここ
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
│   └── dqn_agent.py        # DQNエージェント（実装済み）
├── ui/
│   └── cli_ui.py           # CLI表示（現在はWebUIがメイン）
├── web/
│   ├── app.py              # Flask サーバー（weights があれば DQNAgent を自動ロード）
│   └── templates/
│       └── index.html      # ブラウザUI（アニメーション付き）
├── game_runner.py          # ターン管理（env・agent・UIを繋ぐ）
├── train.py                # 学習スクリプト
├── play.py                 # CLIで人間 vs ランダムAI
└── CLAUDE.md               # このファイル
```

## Webアプリの起動方法
```bash
cd c:\Users\peinn\OneDrive\sandbox\claude-sandbox\RL_ReinforcementLearning\4moku
.venv\Scripts\python web/app.py
# ブラウザで http://localhost:5000 を開く
```
- `weights/dqn_connect4.npz` があれば自動で DQNAgent を読み込む
- なければ RandomAgent にフォールバック（無学習で遊びたい場合は weights を削除 or 移動）

## 学習の実行
```bash
.venv\Scripts\python train.py
.venv\Scripts\python train.py --episodes 20000  # エピソード数指定
```
- 完了後 `weights/dqn_connect4.npz` に保存される

## 設計の重要ポイント

### env/connect4_env.py
- `step()` は相手を呼び出さない（self-playやカリキュラム学習に対応するため）
- `get_state(perspective)` で視点を切り替えられる（DQN訓練・self-playで重要）
- 報酬は常に PLAYER1 視点で返し、`GameRunner` が各エージェント視点に変換する
- 状態は shape `(3, 6, 7)` の float32 配列
  - ch0: 自分のコマ
  - ch1: 相手のコマ
  - ch2: 自分のターンなら 1.0、そうでなければ 0.0

### agents/dqn_agent.py
- 入力: `3 × 6 × 7 = 126`、出力: `7`（列インデックス）
- アーキテクチャ: `Linear(126→256) → ReLU → Linear(256→256) → ReLU → Linear(256→7)`
- NumPyベース（PyTorch不使用）
- ε-greedy + 無効手マスキング、ReplayBuffer、ターゲットネットワーク付き

### 現在のハイパーパラメータ（train.py）
| パラメータ | 値 | 備考 |
|---|---|---|
| `epsilon_decay` | 0.9995 | ~100kステップでε=0.05に到達 |
| `warmup_steps` | 2000 | バッファを十分蓄積してから学習開始 |
| `target_update_interval` | 200 | ターゲットネットをより頻繁に更新 |
| `buffer_capacity` | 50000 | 多様な経験を保持 |
| `batch_size` | 128 | 安定した勾配 |

### 重要なバグ修正（2026-03-16）
- `_train_step`のターゲットQ値計算を `r + γ*maxQ` → `r - γ*maxQ` に修正
- **理由**: `next_state` は「相手のターン」の状態。相手が最善手を取ると自分に不利なので、符号を反転して自分にとっての損失として計算する必要があった
- **効果**: εが下がるにつれて勝率が悪化するバグが解消（500ep時点: 50.4% → 63.2%）

### 中間報酬（シェーピング）の追加（2026-03-16）
- `env/connect4_env.py` の `step()` に `_shaping_reward()` を追加
- 直前に置いたコマを起点に連数をカウント（2連=±0.05、3連=±0.1）
- 盤面全スキャンではなく置いたマス起点のみなので高速（532 eps/s）
- **効果**: 2000ep時点で 62% → **79%超え**（ep1000〜1500付近でピーク）

### game_runner.py
- agent1 = PLAYER1 (X / 赤)、agent2 = PLAYER2 (O / 橙)
- `GameRunner(env, dqn_agent, random_agent)` でステージ1
- `GameRunner(env, dqn_agent, dqn_agent)` でself-play（同一インスタンスも可）

## これからの展望

### ステージ2: カリキュラム学習（次のステップ）
ep1500付近でピーク（~79%）を迎えた後に低下する → ランダム相手への過学習が原因。
段階的に対戦相手を強化することで汎化性能を上げる：

```
Phase 1: vs ランダムAI          → 勝率 70% 超えたら次へ
Phase 2: vs 弱いDQN（過去の自分）→ 勝率 60% 超えたら次へ
Phase 3: vs 中程度のDQN         → ...
```

実装方針：
- `train.py` に `--opponent-path` オプションを追加し、過去の重みをロードしたDQNを相手にできるようにする
- 勝率を監視し、しきい値を超えたら自動でスナップショットを保存・対戦相手を更新する

### ステージ3: Self-play
`GameRunner(env, dqn_agent, dqn_agent)` の構成で実現可能（設計済み）。
カリキュラム学習で汎化性能が上がってからが本番。
