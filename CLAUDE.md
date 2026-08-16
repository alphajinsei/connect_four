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

---

## これからやりたいこと（2026-08-16 策定）

### ③ 3moku を Hugging Face Spaces で一般公開

**目的**: 学習した3moku DQNを誰でも遊べる形でネット公開する。
最終的に https://alphajinsei.com/ のリンクの一つに加える。

**方式: ONNX + Cloudflare Pages（採用・実装済み）**

当初は HF Spaces（Flask）を予定していたが、**Docker/Gradio SDK が有料化されていた**
（PRO月額$9が必要。無料で選べるのは Static のみ）。
Static は静的ファイルしか置けずPythonが動かないため、結局ONNX化が必要になる。
それなら独自ドメインが使える Cloudflare Pages の方が良い、という判断で方針変更。

選定理由:
1. **完全無料**（Cloudflare Pages 無料枠）
2. **独自ドメインが使える** — alphajinsei.com のサブドメインにできる
3. **既存2サイトと同じ運用**（Cloudflare Pages + GitHub連携）
4. サーバーレスなのでスリープも障害もない。推論はブラウザ内で1ms以下

**モデルの規模（公開方式の検討で実測）:**
| | 値 |
|---|---|
| 重みファイル | 1.7 MB |
| パラメータ数 | 430,533 |
| 1手の推論 | 0.85 ms |

**ONNXとは:**
Open Neural Network Exchange = NNの構造と重みを表現する共通ファイル形式。
`.onnx` に変換すると **ONNX Runtime Web**（JSライブラリ）がブラウザ内で推論を実行できる。
「PyTorchが不要になる」のではなく「サーバー側のPyTorchが不要になり、
ブラウザ側のONNX Runtimeが担う」= **実行場所が移動するだけ**。

**公開構成（`web-onnx/`）:**
```
web-onnx/
├── index.html    ← UI（3moku/web/templates/index.html がベース）+ 解説文
├── game.js       ← ゲームロジック + 推論（Python版からの移植）
├── model.onnx    ← 学習済み重み（1.7MB）
└── _headers      ← Cloudflare Pages のキャッシュ設定
```
- 対戦相手は **DQNのみ**（ルールベースAIは移植しない）
- **`3moku/` の学習コードには一切手を加えていない。** 公開用のビューを別途作った
- 変換は `3moku/export_onnx.py`。再学習したら再実行する

**検証済み（2026-08-16）:**
- ONNX変換の同一性: **2000局面で選択手 100% 一致**（export_onnx.py が自動検証、不一致ならexit 1）
- ブラウザ実機（Playwright）: 対局の棋譜が PyTorch 版と完全一致 `[2,0,2,0,2]`、JSエラーなし

**ハマりどころ（記録）:**
- torch 2.10 の `torch.onnx.export` は重みを `model.onnx.data` に分離出力する。
  Web配信で2ファイル構成は事故のもとなので、`save_as_external_data=False` で単一ファイルに統合している
- `torch.onnx.export` には `onnxscript` が別途必要
- `.venv/Scripts/pip` を直接叩くとこの環境では無音で失敗する。`python -m pip` を使うこと

**残作業:**
- Cloudflare Pages へのデプロイ（Build output directory = `web-onnx`、ビルドコマンドなし）
- alphajinsei.com からのリンク追加

### ③-b HF Models でのモデル公開（案D・後回し）

HF Spaces は有料だったが、**Models（モデルのマーケットプレイス本体）は無料**。
「自分のモデルをHFに公開する」体験はこちらで得られる。
アカウント: https://huggingface.co/alphajinsei

Cloudflare Pages 版とは**独立**した構成にする（HFから動的に読むのではなく、
同じ model.onnx を両方に置き、相互リンクするだけ）。理由: CORS、HF側都合での破損、速度。

### ④ ブログ記事化

**目的**: DQNが何をやっているのかを基礎から解説し、自身の理解を整理する。

**書きたい論点（一連の実験で得られたもの）:**
1. 強化学習の枠組み（agent / 環境 / 状態・行動・報酬）
2. Q学習と表の限界 → DQN = 表をNNで近似
   - ただし入力は状態のみ、出力が全行動のQ値（行動は入力しない）
3. 正解ラベルがない問題をどう解くか → ベルマン方程式でブートストラップ
   - 終局の報酬という唯一の真実が、後ろから前へ滲み出す
   - だからゲーム長が効く（4moku失敗の説明になる）
4. 安定化の2工夫 → ターゲットネットワーク（動く的問題）、経験再生（相関問題）
5. **今回の発見**: DQNは「相手が連れて行ってくれる局面」でしか学習できない
   - 経験再生のバッファも、訪問局面が11種類なら意味がない
6. **評価指標の落とし穴**（2種類）
   - 決定論的な条件での勝率は指標にならない（同じ100%が実力66%と83%を隠す）
   - サンプル数不足でも騙される（200戦の「99%」が400戦では81.8%）

## 開発方針
- 各サブプロジェクトは独立して動作する（共通ライブラリは作らない）
- weights/ は各サブプロジェクト内に配置、.gitignore で除外
- サブプロジェクト固有の詳細は各自の CLAUDE.md に記載
