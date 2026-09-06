# Connect Three — ブラウザ版（ONNX）

`3moku/` で学習したDQNを、**ブラウザ内で推論する形**に変換した公開用フロントエンド。
Cloudflare Pages（静的ホスティング）にそのまま置ける。

## 構成

```
web-onnx/
├── index.html         ← UI（3moku/web/templates/index.html がベース）
├── game.js            ← ゲームロジック + 推論（Python版からの移植）
├── model.onnx         ← Lv.3 学習完了後の重み（1.7MB）
├── model_ep4000.onnx  ← Lv.2 学習4,000回時点のスナップショット
├── model_ep1000.onnx  ← Lv.1 学習1,000回時点のスナップショット
├── _headers           ← Cloudflare Pages のキャッシュ設定
└── README.md          ← このファイル
```

サーバーは不要。`.onnx` を **ONNX Runtime Web** がブラウザ内で実行する。

## 対戦相手のレベル

ブログ記事とセットで「学習が進むと強くなる」ことを体験してもらうため、
**学習途中のスナップショットを対戦相手として選べる**ようにしている。

| レベル | 中身 | モデル |
|---|---|---|
| Lv.0 学習前 | ランダムに打つだけ（`RandomAgent`、JS実装） | なし |
| Lv.1 学習1,000回 | ep1000 スナップショット | `model_ep1000.onnx` |
| Lv.2 学習4,000回 | ep4000 スナップショット | `model_ep4000.onnx` |
| Lv.3 学習20,000回 | 最終重み（既定） | `model.onnx` |

モデルは**選択された時点で初めて読み込む**（遅延ロード）。初期表示は Lv.3 のみを読む。

なお、公開版はすべて ε=0（greedy）で動かす。Lv.1 は「学習1,000回時点の重みを
本気で使ったらどうなるか」であり、ε=0.89 だった当時の弱さそのものではない。

## Q値を画面に出さない理由（判断の記録）

一度「各列のQ値をバーで表示する」機能を実装したが、**採用しなかった**。

技術的な問題が1つ、UX上の問題が1つあった。

1. **視点の問題**: このDQNは先手固定・「自分が今から打つ局面」だけを学習しており、
   状態のch2（手番フラグ）は学習時は常に1。人間の手番の盤面は ch2=0 で学習範囲外であり、
   そこにQ値を求めても意味のある値にならない（実際その値に従って打つと負ける）。
   → AIが自分の手番で出した値だけを表示する方式に直せば、ここは解決できる。

2. **配置の問題（こちらが決定打）**: 盤面の各列の上にバーが並ぶと、
   注釈で「これはあなたの指針ではない」と書いても
   **「この列を選べば勝てる」と読めてしまう**。注釈で打ち消せる類の誤読ではない。

Q値の説明はブログ記事側で十分に行っているため、UIからは削除した。

## 重みを更新する手順

`3moku/` で再学習したら、以下を実行して `model.onnx` を作り直す。

```bash
# Lv.3（最終重み）
PYTHONIOENCODING=utf-8 .venv/Scripts/python 3moku/export_onnx.py

# Lv.1 / Lv.2（学習途中のスナップショット）
PYTHONIOENCODING=utf-8 .venv/Scripts/python 3moku/export_onnx.py   --weights weights/snapshots/ep1000_open76_rb72_rand92pct_20260816_124656.pt   --out web-onnx/model_ep1000.onnx
```

このスクリプトは変換だけでなく、
**PyTorch版と同じ手を選ぶかを2000局面で自動検証する**（不一致なら exit 1）。

> Windows注意: `torch.onnx.export` が絵文字を出すため、`PYTHONIOENCODING=utf-8`
> を付けないと cp932 のコンソールで `UnicodeEncodeError` になる。

> 注意: `index.html` に書いてある勝率などの数値は手動更新が必要。
> 数値は `3moku/eval_levels.py`（400戦で実測）で取り直すこと。

## ローカルで動作確認

```bash
cd web-onnx
python -m http.server 8080
# http://127.0.0.1:8080 を開く
```

`file://` では `fetch` が CORS で弾かれるため、簡易サーバー経由で開くこと。

## Cloudflare Pages へのデプロイ

このディレクトリを公開ルートに指定する。ビルドコマンドは不要（静的ファイルのみ）。

| 設定項目 | 値 |
|---|---|
| Build command | （空欄） |
| Build output directory | `web-onnx` |

## 検証済みの動作

- ONNX変換の同一性: 2000局面で選択手 100% 一致（`export_onnx.py` が検証）
- ブラウザ実機: 対局の棋譜が PyTorch 版と完全一致（`[2,0,2,0,2]`）
- JSエラー: なし
