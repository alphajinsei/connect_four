# Connect Four 強化学習プロジェクト

## ユーザーの目標
強化学習を勉強中。Connect系ゲーム（N目並べ）を題材に、AIが学習して強くなる過程を実感したい。
**最終目標: 初心者の人間より強くなる**

## リポジトリ構成

```
(リポジトリルート)/
├── .venv/              ← Python仮想環境（共有）
├── 4moku/              ← Connect Four（6×7盤面、4目並べ）DQN学習
│   ├── CLAUDE.md       ← 4moku固有の詳細（学習ログ・方針・失敗履歴）
│   ├── env/            ← ゲームロジック
│   ├── agents/         ← エージェント群（DQN, RuleBased等）
│   ├── web/            ← Flask WebUI
│   ├── weights/        ← 学習済み重み（.gitignore）
│   ├── train.py        ← 学習スクリプト
│   └── ...
├── 3moku/              ← Connect Three（5×5盤面、3目並べ）DQN学習【新規予定】
├── alphazero/          ← AlphaZero方式（MCTS+NN）でConnect Four
└── CLAUDE.md           ← このファイル（横断）
```

## 実行環境
- Python仮想環境: `.venv/`（リポジトリルートに配置、全サブプロジェクト共有）
- **Pythonの実行は必ず `.venv` を使うこと**
  - 実行: `.venv/Scripts/python`
  - pip: `.venv/Scripts/pip`
- インストール済みパッケージ: `numpy`, `flask`, `torch`（CPU版）

## 各サブプロジェクトの実行方法

### 4moku（Connect Four）
```bash
# 学習
.venv/Scripts/python 4moku/train.py --episodes 30000

# WebUI
.venv/Scripts/python 4moku/web/app.py
```
詳細は `4moku/CLAUDE.md` を参照。

### 3moku（Connect Three）
```bash
# 学習
.venv/Scripts/python 3moku/train.py --episodes 10000

# WebUI
.venv/Scripts/python 3moku/web/app.py
```
詳細は `3moku/CLAUDE.md` を参照。

### alphazero（AlphaZero Connect Four）
```bash
# 学習（推奨設定）
.venv/Scripts/python alphazero/train.py --iterations 100 --games-per-iter 200 --train-steps 100

# 途中から再開（PCスリープ等で止まった場合）
.venv/Scripts/python alphazero/train.py --resume --iterations 100 --games-per-iter 200 --train-steps 100 --eval-interval 5

# WebUI（ポート5001）
.venv/Scripts/python alphazero/web/app.py
```
詳細は `alphazero/CLAUDE.md` を参照。

## 今後の方針（2026-03-19 策定）

### ① 3moku: Connect Three（5×5盤面）でDQN学習 — 完了
- **目的**: DQNが対戦ゲームでどこまで通用するかを確認する
- **結果**: DQNは3mokuで**人間に勝てるレベル**に到達。4,000エピソード（数分）で実質的な最適戦略を獲得
- **方針**: CNN + 中間報酬なし + 強化版RuleBased直接対戦
- **重要な知見**:
  - DQNの適用限界は「状態空間の大きさ」と「ゲーム長」で決まる。3moku（状態空間~10^6）は通用、4moku（~10^12）は通用しない
  - DQNの強さは対戦相手の強さに制約される。対戦相手を強化しなければDQNも強くならない
  - 評価指標は複数持つべき（vs RuleBased + vs Random）。単一指標100%は汎化性能を保証しない

### ② alphazero: AlphaZero方式でConnect Four — ステージ2学習中
- **目的**: 最先端の手法（MCTS + ニューラルネット）への理解を深める
- **背景**: DQNの限界（マルコフ性、単一方策）を体感した上で、AlphaZeroがなぜそれを解決できるかを学ぶ
- **方針**: Connect Four（6×7, 4目）をAlphaZero方式で実装。軽量版（シミュレーション50回/手）でCPU環境でも学習可能
- **パラメータ数**: 301,402（4moku DQN CNN-Aの710Kより少ない。残差ブロック4層+64チャンネルの軽量設計）
- **DQNとの対比**: `alphazero/CLAUDE.md` にDQNの3つの限界とAlphaZeroの解決策を詳述
- **進捗**:
  - ステージ1（games=10, steps=10）: 100iter完走したが vs Random 23%, vs RuleBased 0% で失敗。データ量不足が原因
  - ステージ2（games=200, steps=100）: パラメータ20倍増で再挑戦中。チェックポイント機能追加（`--resume`で途中再開可能）
- **重要な知見**: AlphaZeroの正のフィードバックループを回すには、1イテレーションあたりのデータ量に臨界量がある。10ゲーム/iterでは不足

### 学びの流れ
```
4moku（DQNの限界を体感）
  → 3moku（DQNが通用する範囲を確認）
  → alphazero（DQNの限界を超える手法を学ぶ）
```

## 開発方針
- 各サブプロジェクトは独立して動作する（共通ライブラリは作らない）
- weights/ は各サブプロジェクト内に配置、.gitignore で除外
- サブプロジェクト固有の詳細は各自の CLAUDE.md に記載
