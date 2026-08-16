import random

from env.connect3_env import Connect3Env


class GameRunner:
    def __init__(self, env, agent1, agent2, renderer=None):
        """
        agent1: PLAYER1 (X)
        agent2: PLAYER2 (O)
        renderer: CLIRenderer など（None なら無表示）
        """
        self.env = env
        self.agents = {
            Connect3Env.PLAYER1: agent1,
            Connect3Env.PLAYER2: agent2,
        }
        self.renderer = renderer

    def _random_opening(self, max_plies):
        """
        空盤面からランダムな合法手を 0〜max_plies 手打ち、開始局面を散らす。

        訪問局面を広げるのが目的。実測では max_plies=4 まで開幕直後に終局する
        ことはないが（k=5 で 10.5%、k=6 で 14.9%）、念のため終局したらリセット
        してやり直す。手番は必ず PLAYER1 に戻す（DQN は先手固定で学習するため、
        偶数手だけ打つ）。

        開幕手は env に直接打ち込み、on_step_end を呼ばない。ランダムな手を
        「エージェントが選んだ手」として学習させないため。
        """
        if max_plies <= 0:
            return

        for _ in range(20):  # 終局してしまったらやり直す
            # PLAYER1 の手番を保つため偶数手のみ
            plies = random.randrange(0, max_plies + 1, 2)
            if plies == 0:
                return
            self.env.reset()
            for _ in range(plies):
                self.env.step(random.choice(self.env.get_valid_actions()))
                if self.env.done:
                    break
            if not self.env.done:
                return
        # 20回試して全部終局した場合は空盤面で開始
        self.env.reset()

    def run_episode(self, random_opening_plies=0):
        state = self.env.reset()
        self._random_opening(random_opening_plies)
        for agent in self.agents.values():
            agent.on_episode_start()

        if self.renderer:
            self.renderer.render(self.env)

        total_rewards = {Connect3Env.PLAYER1: 0.0, Connect3Env.PLAYER2: 0.0}
        steps = 0
        p1_states = []  # PLAYER1 が実際に手を選んだ局面（訪問局面数の計測用）

        pending = {
            Connect3Env.PLAYER1: None,
            Connect3Env.PLAYER2: None,
        }

        while not self.env.done:
            current_player = self.env.current_player
            agent = self.agents[current_player]
            valid_actions = self.env.get_valid_actions()

            if current_player == Connect3Env.PLAYER1:
                p1_states.append(self.env.board.tobytes())

            agent_state = self.env.get_state(perspective=current_player)

            if pending[current_player] is not None:
                p = pending[current_player]
                p["agent"].on_step_end(
                    p["state"], p["action"], p["reward"], agent_state, False
                )
                total_rewards[current_player] += p["reward"]
                pending[current_player] = None

            if self.renderer and hasattr(agent, 'is_human') and agent.is_human:
                action = self.renderer.prompt_human_action(valid_actions)
            else:
                action = agent.get_action(agent_state, valid_actions)

            next_state_raw, reward_p1, done, info = self.env.step(action)

            reward_for_agent = reward_p1 if current_player == Connect3Env.PLAYER1 else -reward_p1

            if done:
                next_agent_state = self.env.get_state(perspective=current_player)
                agent.on_step_end(agent_state, action, reward_for_agent, next_agent_state, True)
                total_rewards[current_player] += reward_for_agent

                opponent = Connect3Env.PLAYER2 if current_player == Connect3Env.PLAYER1 else Connect3Env.PLAYER1
                if pending[opponent] is not None:
                    p = pending[opponent]
                    terminal_reward = reward_p1 if opponent == Connect3Env.PLAYER1 else -reward_p1
                    combined_reward = p["reward"] + terminal_reward
                    opp_next_state = self.env.get_state(perspective=opponent)
                    p["agent"].on_step_end(
                        p["state"], p["action"], combined_reward, opp_next_state, True
                    )
                    total_rewards[opponent] += combined_reward
                    pending[opponent] = None
            else:
                pending[current_player] = {
                    "agent": agent,
                    "state": agent_state,
                    "action": action,
                    "reward": reward_for_agent,
                }

            steps += 1

            if self.renderer:
                self.renderer.render(self.env)

        if self.renderer:
            self.renderer.show_result(self.env.winner)

        return {
            "winner": self.env.winner,
            "steps": steps,
            "reward_p1": total_rewards[Connect3Env.PLAYER1],
            "reward_p2": total_rewards[Connect3Env.PLAYER2],
            "p1_states": p1_states,
        }
