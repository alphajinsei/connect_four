# Connect Four 強化学習プロジェクト

## ユーザーの目標
強化学習（DQN）を勉強中。Connect Four（四目並べ）を題材に、AIが学習して強くなる過程を実感したい。

## 学習ロードマップ
段階的に以下の順で進める：
1. **相手を固定**（ランダムエージェント） ← 完了（勝率 ~79% 達成）
2. **カリキュラム学習**（段階的に強い相手と対戦） ← 完了（全3フェーズ通過）
3. **Self-play**（自分自身と対戦して強化） ← 次のステップ

## 実行環境
- Python仮想環境: `.venv/`（プロジェクトルートに配置）
- **Pythonの実行は必ず `.venv` を使うこと**
  - 実行: `.venv/Scripts/python`
  - pip: `.venv/Scripts/pip`
- インストール済みパッケージ: `numpy`, `flask`
- 学習速度: **約1エピソード/秒**（1000ep≒17分、10000ep≒2.8時間）

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
├── weights/
│   ├── dqn_connect4.npz    # 学習済み重み（学習完了後に保存）
│   ├── train_log.txt       # 学習ログ（リアルタイム書き出し）
│   └── snapshots/          # カリキュラム学習フェーズ移行時のスナップショット
├── game_runner.py          # ターン管理（env・agent・UIを繋ぐ）
├── train.py                # 学習スクリプト（カリキュラム・self-play対応済み）
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
- なければ RandomAgent にフォールバック

## 学習の実行
```bash
# ステージ1: ランダム相手で学習
.venv\Scripts\python train.py

# ステージ2: カリキュラム学習
.venv\Scripts\python train.py --curriculum --episodes 15000

# 手動で特定スナップショットを相手に指定
.venv\Scripts\python train.py --opponent-path weights/snapshots/snapshot_01_ep2000

# エピソード数・評価間隔を指定
.venv\Scripts\python train.py --episodes 20000 --eval-interval 500
```
- 完了後 `weights/dqn_connect4.npz` に保存される
- 学習ログは `weights/train_log.txt` にリアルタイム書き出しされる

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
| `epsilon_decay` | 0.99990 | ~30000ステップ(≒3750ep)でε=0.05に到達 |
| `warmup_steps` | 2000 | バッファを十分蓄積してから学習開始 |
| `target_update_interval` | 200 | ターゲットネットをより頻繁に更新 |
| `buffer_capacity` | 50000 | 多様な経験を保持 |
| `batch_size` | 128 | 安定した勾配 |

**epsilon_decayの調整履歴:**
- `0.9995`（旧）: warmup後750epでε=0.05に到達 → 探索が早期終了して過学習
- `0.99990`（現行）: 3750epかけてゆっくり探索 → ep1000〜2500の勝率が64%→68〜69%に改善

### カリキュラム学習の設定（train.py）
| フェーズ | 相手 | 昇格しきい値 |
|---|---|---|
| Phase 1 | ランダムAI | 勝率 **65%** 超え |
| Phase 2 | 弱いDQN（Phase1卒業スナップショット） | 勝率 **60%** 超え |
| Phase 3 | 中程度のDQN（Phase2卒業スナップショット） | 勝率 **60%** 超え |

- 昇格時に `weights/snapshots/snapshot_NN_epXXXX.npz` を自動保存
- Phase 2以降の相手のεは `0.05`（少しランダム性を残す）
- ログは `weights/train_log.txt` にリアルタイム書き出し

### 重要なバグ修正（2026-03-16）
- `_train_step`のターゲットQ値計算を `r + γ*maxQ` → `r - γ*maxQ` に修正
- **理由**: `next_state` は「相手のターン」の状態。相手が最善手を取ると自分に不利なので、符号を反転して自分にとっての損失として計算する必要があった
- **効果**: εが下がるにつれて勝率が悪化するバグが解消

### 中間報酬（シェーピング）の追加（2026-03-16）
- `env/connect4_env.py` の `step()` に `_shaping_reward()` を追加
- 直前に置いたコマを起点に連数をカウント（2連=±0.05、3連=±0.1）
- 盤面全スキャンではなく置いたマス起点のみなので高速（~1ep/s を維持）
- **効果**: 勝率 62% → **79%超え**

### game_runner.py
- agent1 = PLAYER1 (X / 赤)、agent2 = PLAYER2 (O / 橙)
- `GameRunner(env, dqn_agent, random_agent)` でステージ1・2
- `GameRunner(env, dqn_agent, dqn_agent)` でself-play（同一インスタンスも可・設計済み）

## ステージ2の学習結果と知見（2026-03-16完了）

### カリキュラム進行ログ
```
Phase 1: vs ランダム
  ep500:  45.2% (ε=0.77)   ← 探索中
  ep1000: 64.6% (ε=0.49)
  ep1500: 59.8% (ε=0.31)   ← 揺れあり
  ep2000: 65.6% (ε=0.19)   → 65%達成・Phase2へ

Phase 2: vs snap01 (ep2000時点の自分)
  ep2500: 66.2% (ε=0.12)   → 60%達成・Phase3へ

Phase 3: vs snap02 (ep2500時点の自分)
  ep3000: 67.6% (ε=0.07)   → 60%達成・全フェーズ完了

フェーズ完了後（snap02固定）:
  ep3500: 71.6% (ε=0.05)   ← ピーク
  ep4000: 64.2%             ↓ 過学習開始
  ep4500: 63.4%
  ep5000: 61.2%
```

### ステージ2で判明した課題：「固定相手への過学習」
- フェーズ完了後も相手が固定されるため、ε収束後にまた過学習が始まる
- ステージ1（vs ランダム固定）と全く同じパターン
- **根本解決策: Self-play**（相手=自分自身なので常に同レベルの相手と戦い続ける）

## これからの展望

### ステージ3: Self-play（次のステップ）
`GameRunner(env, dqn_agent, dqn_agent)` の構成で実現可能（設計済み）。

**実装方針:**
- 同一の DQNAgent インスタンスを PLAYER1・PLAYER2 両方に渡す
- 一定間隔（例: 500ep）で自分のスナップショットを保存し、相手側を更新する
  （完全な同一インスタンスだと学習が不安定になる可能性があるため）
- あるいは相手側は完全固定にして、一定ep後に手動で更新するシンプルな実装から始める

**期待される効果:**
- 相手が常に「今の自分」なので、特定パターンへの過学習が起きにくい
- 強くなるにつれて相手も強くなるため、学習が継続しやすい
