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
├── alphazero/          ← AlphaZero方式の実装【新規予定】
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

### alphazero
TODO: 実装予定

## 今後の方針（2026-03-19 策定）

### ① 3moku: Connect Three（5×5盤面）でDQN学習 — 進行中
- **目的**: DQNが対戦ゲームでどこまで通用するかを確認する
- **背景**: Connect Four（6×7, 4目）はDQN単体では限界があった（10ステージの試行錯誤で確認済み）。盤面を小さくし目標を3目に下げることで、DQN単体でも学習が成功する可能性が高い
- **方針**: 4moku/のコードをベースに、盤面サイズとconnect数を変更。CNN + 中間報酬なしの方針は維持
- **進捗**: DQNは簡易RuleBasedに100%勝利達成。ただし人間には負ける→RuleBasedを強化して再学習中
- **重要な知見**: DQNの強さは対戦相手の強さに制約される。対戦相手を強化しなければDQNも強くならない

### ② alphazero: AlphaZero方式でConnect Four
- **目的**: 最先端の手法（MCTS + ニューラルネット）への理解を深める
- **背景**: DQNの限界（マルコフ性、単一方策）を体感した上で、AlphaZeroがなぜそれを解決できるかを学ぶ
- **方針**: Connect Four（6×7, 4目）をAlphaZero方式で実装。軽量版（シミュレーション50回/手、数千ゲーム）でCPU環境でも数時間で学習可能な見込み
- **優先度**: ①の後に着手。まず理論の理解から始める

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
