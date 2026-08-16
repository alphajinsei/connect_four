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

### ① 3moku: Connect Three（5×5盤面）でDQN学習 — Stage 5 完了
- **目的**: DQNが対戦ゲームでどこまで通用するかを確認する
- **方針**: CNN + 中間報酬なし + 確率的RuleBased直接対戦 + ランダム初期局面
- **経緯**:
  - Stage 4 までで vs RuleBased 100% に到達し「人間に勝てるレベル」と結論づけたが、**これは誤りだった**（2026-08-16 に人間との対戦で判明）
  - 実測すると、300ゲームで棋譜は2通り・DQNが遭遇する盤面はわずか**11種類**（状態空間 ~6×10^6 の 0.0002%）。100%は「11局面の暗記テスト満点」だった
  - 同じ重みでも開幕をランダム化すると **100% → 73%** に低下
  - Stage 5: ランダム初期局面（k≤4）+ 確率的相手（noise=0.15）で訪問局面を拡大し、評価指標を「ランダム開幕からの勝率」に刷新。20,000ep ゼロ学習で完走
  - **Stage 5 結果（400戦で測定）: 汎化性能（乱開幕）66.2% → 83.2%、訪問局面 11 → 4,477（407倍）**。ただし「固定RB 100%」は新旧どちらも同じで、従来指標では実力差を区別できなかった
  - **人間との対戦で強くなったことを確認し、3moku の学習は完成とした（2026-08-16）**
  - 実戦条件（空盤面）での勝率は vs 決定論RB 100%、vs 確率的RB 93.8%。乱開幕の83.2%は汎化性能の指標であり、実戦の強さとは別物（開幕4手がランダムなため不利な局面も含む）
- **重要な知見**:
  - DQNの適用限界は「状態空間の大きさ」と「ゲーム長」で決まる。3moku（状態空間~10^6）は通用、4moku（~10^12）は通用しない
  - **DQNの天井を決めるのは「相手の強さ」ではなく「相手が到達する局面の集合」**。決定論的な相手は決定論的な1本道しか作らないため、相手をいくら強化しても訪問局面は増えない（Stage 3 で RuleBased を強化しても結果が変わらなかった理由）
  - **決定論的な条件での勝率は指標として機能しない**。同じ重みが 100% と 66% の両方を示しうる
  - 訪問局面数を必ずログに出す。これが学習の健全性を測る一次指標
  - **評価のサンプル数不足も指標を壊す**。200戦では「99%」に見えた断面が400戦では81.8%で、ベスト断面の判定自体も誤っていた。結論を出す際は400戦以上で測り直す

### ② alphazero: AlphaZero方式でConnect Four — ステージ2学習中
- **目的**: 最先端の手法（MCTS + ニューラルネット）への理解を深める
- **背景**: DQNの限界（マルコフ性、単一方策）を体感した上で、AlphaZeroがなぜそれを解決できるかを学ぶ
- **方針**: Connect Four（6×7, 4目）をAlphaZero方式で実装。軽量版（シミュレーション50回/手）でCPU環境でも学習可能
- **パラメータ数**: 301,402（4moku DQN CNN-Aの710Kより少ない。残差ブロック4層+64チャンネルの軽量設計）
- **DQNとの対比**: `alphazero/CLAUDE.md` にDQNの3つの限界とAlphaZeroの解決策を詳述
- **進捗**:
  - ステージ1（games=10, steps=10）: 100iter完走したが vs Random 23%, vs RuleBased 0% で失敗。データ量不足が原因
  - ステージ2（games=200, steps=100）: パラメータ20倍増で再挑戦したが **2026-03-27 に100iter完走して再び失敗**。vs Random 20〜33%、vs RuleBased ほぼ0%
    - P-Loss が 1.86 → 1.79 とほとんど下がっていない（7列の一様分布 = ln(7) ≈ 1.95。方策がほぼランダムから抜け出せていない）
    - self-play の勝敗が P1:104 / P2:79 / D:17 前後で安定。先手必勝のConnect Fourにしては先手勝率が低すぎ、質の低い対局が続いている
    - **データ量の問題ではなかった可能性が高い。** 次はパラメータ増強より学習ループのバグ切り分け（πの正規化、valueの符号反転、stateの視点変換）を優先すべき
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
