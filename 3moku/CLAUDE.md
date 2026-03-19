# Connect Three (3目並べ) DQN学習プロジェクト

## 概要
5×5盤面の Connect Three（重力付き3目並べ）を DQN で学習する。
4moku/ のコードをベースに、盤面サイズとconnect数を変更。

## 構成
```
3moku/
├── CLAUDE.md       ← このファイル
├── env/            ← ゲームロジック（Connect3Env: 5×5, 3目）
├── agents/         ← エージェント群（DQN, RuleBased等）
├── web/            ← Flask WebUI（ポート5001）
├── weights/        ← 学習済み重み（.gitignore）
├── train.py        ← 学習スクリプト
├── play.py         ← CLI対戦
└── game_runner.py  ← ゲーム進行管理
```

## 4mokuからの変更点
- 盤面: 6×7 → 5×5
- 勝利条件: 4連 → 3連
- CNN flatten: 64*6*7=2688 → 64*5*5=1600
- アクション数: 7 → 5
- カリキュラム: 5フェーズ → 4フェーズ（盤面が小さいため簡略化）
- バッファ容量: 200000 → 100000（ゲームが短いため）
- WebUIポート: 5000 → 5001（4mokuと同時起動可能）

## 実行方法
```bash
# 学習
.venv/Scripts/python 3moku/train.py --episodes 30000

# WebUI
.venv/Scripts/python 3moku/web/app.py

# CLI対戦
.venv/Scripts/python 3moku/play.py
```

## カリキュラム
| Phase | noise | 目標勝率 | 対戦相手 |
|-------|-------|----------|----------|
| 1     | 0.8   | 70%      | ほぼランダム |
| 2     | 0.5   | 70%      | 半分ルールベース |
| 3     | 0.2   | 60%      | 8割ルールベース |
| 4     | 0.0   | 50%      | 完全ルールベースAI |

## 学習ログ
（学習実行後に記録）
