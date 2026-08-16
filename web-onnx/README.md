# Connect Three — ブラウザ版（ONNX）

`3moku/` で学習したDQNを、**ブラウザ内で推論する形**に変換した公開用フロントエンド。
Cloudflare Pages（静的ホスティング）にそのまま置ける。

## 構成

```
web-onnx/
├── index.html    ← UI（3moku/web/templates/index.html がベース）
├── game.js       ← ゲームロジック + 推論（Python版からの移植）
├── model.onnx    ← 学習済み重み（1.7MB、export_onnx.py が生成）
├── _headers      ← Cloudflare Pages のキャッシュ設定
└── README.md     ← このファイル
```

サーバーは不要。`model.onnx` を **ONNX Runtime Web** がブラウザ内で実行する。

## なぜ ONNX なのか

Cloudflare Pages は静的ファイルしか置けず、Python/PyTorch が動かない。
ONNX は「NNの構造と重みを表現する共通ファイル形式」で、
ブラウザ側の ONNX Runtime Web (JS) が読んで推論できる。

つまり「PyTorchが不要になる」のではなく、
**サーバー側のPyTorchが不要になり、ブラウザ側のONNX Runtimeが担う**（実行場所が移動する）。

## 重みを更新する手順

`3moku/` で再学習したら、以下を実行して `model.onnx` を作り直す。

```bash
.venv/Scripts/python 3moku/export_onnx.py
```

このスクリプトは変換だけでなく、
**PyTorch版と同じ手を選ぶかを2000局面で自動検証する**（不一致なら exit 1）。

> 注意: `index.html` に書いてある勝率などの数値は手動更新が必要。

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
