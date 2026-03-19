"""
AlphaZero ニューラルネットワーク（デュアルヘッド: Policy + Value）

入力: (B, 3, 6, 7) — 盤面の3チャンネル表現
出力:
  policy: (B, 7) — 各列に打つ確率（log-softmax前の logits）
  value:  (B, 1) — 現在のプレイヤーの勝率予測（-1.0〜+1.0、tanh）

アーキテクチャ:
  共有部: Conv(3→64, 3×3) → BN → ReLU
        → ResBlock(64) × 4
  Policyヘッド: Conv(64→2, 1×1) → BN → ReLU → flatten → Linear → log_softmax
  Valueヘッド:  Conv(64→1, 1×1) → BN → ReLU → flatten → Linear(42→64) → ReLU → Linear(64→1) → tanh

パラメータ数: 約200K（CPU学習に適した軽量設計）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """残差ブロック: Conv → BN → ReLU → Conv → BN → skip connection → ReLU"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class AlphaZeroNet(nn.Module):
    def __init__(self, num_res_blocks=4, channels=64,
                 board_h=6, board_w=7, num_actions=7):
        super().__init__()
        self.board_h = board_h
        self.board_w = board_w
        self.num_actions = num_actions

        # 共有: 入力層 + 残差ブロック
        self.input_conv = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_res_blocks)])

        # Policy ヘッド
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_h * board_w, num_actions)

        # Value ヘッド
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_h * board_w, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # 共有部
        h = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            h = block(h)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # logits (B, num_actions)

        # Value
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # (B, 1), range [-1, 1]

        return p, v.squeeze(-1)

    def predict(self, state):
        """
        単一の状態 (3, H, W) に対して (policy, value) を返す。
        policy: (num_actions,) 確率分布（softmax済み、違法手込み）
        value: スカラー
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(state).unsqueeze(0).float()
            logits, v = self(x)
            policy = F.softmax(logits, dim=1).squeeze(0).numpy()
            value = v.item()
        return policy, value
