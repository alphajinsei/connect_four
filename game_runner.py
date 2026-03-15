from env.connect4_env import Connect4Env


class GameRunner:
    def __init__(self, env, agent1, agent2, renderer=None):
        """
        agent1: PLAYER1 (X)
        agent2: PLAYER2 (O)
        renderer: CLIRenderer など（None なら無表示）
        """
        self.env = env
        self.agents = {
            Connect4Env.PLAYER1: agent1,
            Connect4Env.PLAYER2: agent2,
        }
        self.renderer = renderer

    def run_episode(self):
        state = self.env.reset()
        for agent in self.agents.values():
            agent.on_episode_start()

        if self.renderer:
            self.renderer.render(self.env)

        total_rewards = {Connect4Env.PLAYER1: 0.0, Connect4Env.PLAYER2: 0.0}
        steps = 0

        while not self.env.done:
            current_player = self.env.current_player
            agent = self.agents[current_player]
            valid_actions = self.env.get_valid_actions()

            # 各エージェントに自分視点の状態を渡す
            agent_state = self.env.get_state(perspective=current_player)

            if self.renderer and hasattr(agent, 'is_human') and agent.is_human:
                action = self.renderer.prompt_human_action(valid_actions)
            else:
                action = agent.get_action(agent_state, valid_actions)

            next_state_raw, reward_p1, done, info = self.env.step(action)

            # PLAYER2 視点では報酬を反転
            reward_for_agent = reward_p1 if current_player == Connect4Env.PLAYER1 else -reward_p1

            next_agent_state = self.env.get_state(perspective=current_player)
            agent.on_step_end(agent_state, action, reward_for_agent, next_agent_state, done)

            total_rewards[current_player] += reward_for_agent
            steps += 1

            if self.renderer:
                self.renderer.render(self.env)

        if self.renderer:
            self.renderer.show_result(self.env.winner)

        return {
            "winner": self.env.winner,
            "steps": steps,
            "reward_p1": total_rewards[Connect4Env.PLAYER1],
            "reward_p2": total_rewards[Connect4Env.PLAYER2],
        }
