"""
モンテカルロ木探索（MCTS） — AlphaZero方式

AlphaZeroのMCTSは以下の4ステップを繰り返す:
  1. Select:   UCBスコアが最大の子ノードを辿って葉ノードまで降りる
  2. Expand:   葉ノードでNNを呼び出し、policy と value を取得、子ノードを作成
  3. Evaluate: NNのvalue予測を使う（ランダムロールアウトは行わない）
  4. Backup:   葉から根までvalueを逆伝播（手番が交互なので符号を反転）

UCB式: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
  - Q(s,a): 行動aの平均value
  - P(s,a): NNのpolicy予測（事前確率）
  - N(s):   親ノードの訪問回数
  - N(s,a): 行動aの訪問回数
  - c_puct: 探索と活用のバランス定数
"""
import math
import numpy as np


class MCTSNode:
    __slots__ = ['parent', 'action', 'prior', 'env_state',
                 'children', 'visit_count', 'value_sum', 'is_expanded']

    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action
        self.prior = prior       # P(s, a): NNの事前確率
        self.env_state = None    # このノードに対応する env の clone
        self.children = {}       # action -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct):
        """UCBスコア = Q + c_puct * P * sqrt(N_parent) / (1 + N)"""
        if self.parent is None:
            return 0.0
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + exploration

    def select_child(self, c_puct):
        """UCBスコアが最大の子ノードを選択"""
        return max(self.children.values(), key=lambda c: c.ucb_score(c_puct))

    def expand(self, env, policy):
        """
        葉ノードを展開: 合法手それぞれに子ノードを作成
        policy: (num_actions,) NNのsoftmax出力
        """
        self.is_expanded = True
        valid_actions = env.get_valid_actions()

        # 合法手のpolicyを正規化
        mask = env.get_valid_actions_mask()
        masked_policy = policy * mask
        policy_sum = masked_policy.sum()
        if policy_sum > 0:
            masked_policy /= policy_sum
        else:
            # NNの出力が全て違法手に集中している場合、均等に
            masked_policy = mask / mask.sum()

        for action in valid_actions:
            child = MCTSNode(parent=self, action=action, prior=masked_policy[action])
            self.children[action] = child

    def backup(self, value):
        """葉からルートまで value を逆伝播（手番交互なので毎回符号反転）"""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # 相手にとっては逆の評価
            node = node.parent


class MCTS:
    def __init__(self, network, num_simulations=50, c_puct=1.5):
        """
        Args:
            network: AlphaZeroNet（predict メソッドを持つ）
            num_simulations: 1手あたりのシミュレーション回数
            c_puct: 探索定数（大きいほど探索寄り）
        """
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, env, add_noise=False):
        """
        MCTSを実行し、ルートノードの行動確率（訪問回数に基づく）を返す。

        Args:
            env: 現在の盤面（変更されない、内部でcloneする）
            add_noise: Trueならルートにディリクレノイズを追加（self-play時）

        Returns:
            action_probs: (num_actions,) 訪問回数に基づく行動確率
        """
        root = MCTSNode()
        root.env_state = env.clone()

        # ルートを展開
        state = root.env_state.get_canonical_state()
        policy, value = self.network.predict(state)
        root.expand(root.env_state, policy)

        # ルートにディリクレノイズを追加（探索の多様性）
        if add_noise and root.children:
            noise = np.random.dirichlet([0.3] * len(root.children))
            for i, child in enumerate(root.children.values()):
                child.prior = 0.75 * child.prior + 0.25 * noise[i]

        # シミュレーション
        for _ in range(self.num_simulations):
            node = root
            sim_env = root.env_state.clone()

            # 1. Select: 展開済みの子を辿る
            while node.is_expanded and node.children:
                node = node.select_child(self.c_puct)
                done, winner = sim_env.step(node.action)
                if done:
                    break

            # 2. Expand & Evaluate
            if sim_env.done:
                # ゲーム終了: 実際の結果を使う
                if winner == 0:
                    value = 0.0
                else:
                    # winnerが「直前に打ったプレイヤー」ならそのプレイヤーにとって+1
                    # nodeの視点 = 打った側の相手 → value = -1
                    value = -1.0  # 直前に打った側が勝ち = 今のノード視点では負け
            else:
                # 葉ノード: NNで評価して展開
                state = sim_env.get_canonical_state()
                policy, value = self.network.predict(state)
                node.expand(sim_env, policy)
                value = -value  # NNは「現在のプレイヤー視点」のvalueを返すが、
                                # backupでは「このノードの親（打った側）視点」が必要

            # 3. Backup
            node.backup(value)

        # 訪問回数 → 行動確率
        action_probs = np.zeros(env.NUM_ACTIONS, dtype=np.float32)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count

        total = action_probs.sum()
        if total > 0:
            action_probs /= total

        return action_probs

    def get_action_with_temperature(self, action_probs, temperature=1.0):
        """
        行動確率から温度付きで行動を選択。

        temperature=1.0: 訪問回数に比例した確率で選択（序盤の多様性）
        temperature→0: 最も訪問された行動を確定的に選択（終盤の精度）
        """
        if temperature < 1e-8:
            # greedy
            return int(np.argmax(action_probs))

        # 温度スケーリング
        probs = action_probs ** (1.0 / temperature)
        total = probs.sum()
        if total < 1e-8:
            valid = (action_probs > 0).astype(np.float32)
            probs = valid / valid.sum()
        else:
            probs /= total

        return int(np.random.choice(len(probs), p=probs))
