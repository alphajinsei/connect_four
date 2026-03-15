# Connect Four 強化学習プロジェクト

四目並べ（Connect Four）を題材に、DQN（Deep Q-Network）による強化学習を実装・実験するプロジェクトです。

## 概要

- **目的**: 強化学習の基礎（DQN）を、Connect Fourで手を動かしながら学ぶ
- **実装**: PyTorchなし、NumPyのみで手書きニューラルネット
- **進捗**: ステージ1（vs ランダムAI）で勝率 **~79%** 達成

## 学習ロードマップ

| ステージ | 内容 | 状態 |
|---------|------|------|
| 1 | vs ランダムAI | 完了（勝率 ~79%） |
| 2 | カリキュラム学習（段階的に強い相手） | 進行中 |
| 3 | Self-play | 未着手 |

## セットアップ

```bash
# 仮想環境の作成・有効化
python -m venv .venv
.venv\Scripts\activate

# 依存パッケージのインストール
.venv\Scripts\pip install numpy flask
```

## 使い方

### 学習

```bash
.venv\Scripts\python train.py
.venv\Scripts\python train.py --episodes 10000 --eval-interval 500
```

完了後、`weights/dqn_connect4.npz` に重みが保存されます。

### ブラウザで対戦

```bash
.venv\Scripts\python web/app.py
```

`http://localhost:5000` をブラウザで開いてください。
`weights/dqn_connect4.npz` があれば自動でDQNエージェントを読み込みます（なければRandomAgent）。

## ファイル構成

```
4moku/
├── env/
│   └── connect4_env.py     # ゲームロジック（重力・勝利判定・状態表現・中間報酬）
├── agents/
│   ├── base_agent.py       # 抽象基底クラス
│   ├── human_agent.py      # 人間プレイヤー（WebUI経由）
│   ├── random_agent.py     # ランダムAI
│   └── dqn_agent.py        # DQNエージェント（NumPy手書き実装）
├── ui/
│   └── cli_ui.py           # CLI表示
├── web/
│   ├── app.py              # Flask サーバー
│   └── templates/
│       └── index.html      # ブラウザUI（アニメーション付き）
├── game_runner.py          # ターン管理（env・agent・UIを繋ぐ）
├── train.py                # 学習スクリプト
├── play.py                 # CLIで人間 vs ランダムAI
├── weights/                # 学習済み重み（.npz）
└── CLAUDE.md               # 設計メモ・開発ログ
```

## DQN実装の概要

### ネットワーク構造

```
入力: (3, 6, 7) → flatten → 126次元
  ch0: 自分のコマ
  ch1: 相手のコマ
  ch2: 自分のターンなら 1.0

Linear(126 -> 256) -> ReLU -> Linear(256 -> 256) -> ReLU -> Linear(256 -> 7)

出力: 各列のQ値 (7次元)
```

### 主な設計ポイント

- **ε-greedy + 無効手マスキング**: 埋まった列は選ばない
- **Experience Replay**: バッファ容量 50,000
- **ターゲットネットワーク**: 200ステップごとに同期
- **中間報酬（シェーピング）**: 2連=±0.05、3連=±0.1（直前に置いたコマ起点で高速計算）
- **ターゲットQ値**: `r - γ * max Q(s')` ※ s'は相手のターンなので符号を反転

### ハイパーパラメータ

| パラメータ | 値 |
|-----------|-----|
| learning rate | 1e-3 |
| gamma | 0.99 |
| epsilon_decay | 0.9995 |
| epsilon_end | 0.05 |
| batch_size | 128 |
| buffer_capacity | 50,000 |
| warmup_steps | 2,000 |
| target_update_interval | 200 |

## 学習結果（ステージ1）

ランダムAI相手に2,000エピソード学習した結果：

```
Ep   500:  67%
Ep  1000:  79%
Ep  1500:  80%
Ep  2000:  69%  ← ランダム相手への過学習が始まる兆候
```

ep1500付近でピークを迎え、その後下がり始めるためステージ2（カリキュラム学習）へ移行予定。
