from Agents.RandomAgent import RandomAgent
from Agents.DQNAgent import DQNAgent

import gym
import numpy as np

class Trainer:

    def __init__(self, args):
        self.env = gym.make(args.env_name)
        self.agent = RandomAgent(self.env)
        self.agent_dqn = DQNAgent(self.env)
        self.n_episode = args.n_episodes
        self.step = 0
        self.loss_val = np.infty
        self.game_length = 0
        self.total_max_q = 0
        self.mean_max_q = 0.0

    def run(self):
        observation = self.env.reset()
        for _ in range(self.n_episode):
            action = self.agent.choose_action(observation)
            observation, reward, done, info = self.env.step(action)
            if done:
                break

    def run_dqn(self):
        observation = self.env.reset()
        observation = self.agent_dqn.preprocessing(observation)
        done = True
        for step in range(self.n_episode):
            print("Training step {}/{} ({:.1f})%\tLoss {:5f}\tMean Max-Q {:5f}   \n".format(
            step, self.n_episode, step * 100 / self.n_episode, self.loss_val, self.mean_max_q), end="")
            if done:
                observation = self.env.reset()
                observation = self.agent_dqn.preprocessing(observation)

            action = self.agent_dqn.choose_action(observation)
            next_state, reward, done, info = self.env.step(action)

            next_state = self.agent_dqn.preprocessing(next_state)
            self.agent_dqn.remember((observation, action, reward,
                next_state, done))

            observation = next_state
            self.total_max_q += self.agent_dqn.max_q_value
            self.game_length += 1
            if done:
                self.mean_max_q = self.total_max_q / self.game_length
                self.total_max_q = 0.0
                self.game_length = 0

            if (self.step%10 == 9):
                self.agent_dqn.train(50)
                self.loss_val = self.agent_dqn.loss_val

            self.step+=1
