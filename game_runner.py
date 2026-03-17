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

        # 遅延コールバック: 各プレイヤーの「前回の手」の情報を保持
        # on_step_end は「相手が1手打った後（再び自分のターン）」に呼ぶ
        # これにより next_state が「自分のターンの状態」になり、
        # 通常の DQN 更新式 Q(s,a) = r + γ*maxQ(s',a') がそのまま使える
        pending = {
            Connect4Env.PLAYER1: None,
            Connect4Env.PLAYER2: None,
        }

        while not self.env.done:
            current_player = self.env.current_player
            agent = self.agents[current_player]
            valid_actions = self.env.get_valid_actions()

            # 各エージェントに自分視点の状態を渡す
            agent_state = self.env.get_state(perspective=current_player)

            # 相手の手番が終わった → 前回の自分の遷移を確定させる
            # next_state = 今の自分視点の状態（相手が打った後、再び自分のターン）
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

            # PLAYER2 視点では報酬を反転
            reward_for_agent = reward_p1 if current_player == Connect4Env.PLAYER1 else -reward_p1

            if done:
                # ゲーム終了: 今手を打ったプレイヤーの遷移を即確定（done=True）
                next_agent_state = self.env.get_state(perspective=current_player)
                agent.on_step_end(agent_state, action, reward_for_agent, next_agent_state, True)
                total_rewards[current_player] += reward_for_agent

                # 相手側に pending があれば、相手も done=True で確定させる
                # 相手の報酬 = 勝敗報酬（中間報酬は自分の手で発生するので、
                # 相手の pending には自分の手の中間報酬が入っている）
                opponent = Connect4Env.PLAYER2 if current_player == Connect4Env.PLAYER1 else Connect4Env.PLAYER1
                if pending[opponent] is not None:
                    p = pending[opponent]
                    # 相手にとっての終局報酬を加算
                    terminal_reward = -reward_p1 if opponent == Connect4Env.PLAYER1 else reward_p1
                    # 実際にはここが少し厄介: pending の reward は「相手が前回打った手の中間報酬」
                    # 終局報酬は「今の手で決着がついた」ことによるもの
                    # → pending の中間報酬 + 終局報酬を合算して返す
                    combined_reward = p["reward"] + terminal_reward
                    opp_next_state = self.env.get_state(perspective=opponent)
                    p["agent"].on_step_end(
                        p["state"], p["action"], combined_reward, opp_next_state, True
                    )
                    total_rewards[opponent] += combined_reward
                    pending[opponent] = None
            else:
                # ゲーム継続: 遷移を保留（相手が打った後に確定させる）
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
            "reward_p1": total_rewards[Connect4Env.PLAYER1],
            "reward_p2": total_rewards[Connect4Env.PLAYER2],
        }
