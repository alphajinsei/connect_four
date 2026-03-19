from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def get_action(self, state, valid_actions):
        """状態と合法手リストから行動を選択して返す。"""

    def on_episode_start(self):
        """エピソード開始時に呼ばれる（任意）。"""

    def on_step_end(self, state, action, reward, next_state, done):
        """ステップ終了時に呼ばれる（学習エージェントが使用）。"""
