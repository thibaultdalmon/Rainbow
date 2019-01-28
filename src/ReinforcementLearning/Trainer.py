from Agents.RandomAgent import RandomAgent
from utils.Plots import plot_rewards

import gym


class Trainer:

    def __init__(self, args):
        self.env = gym.make(args.env_name)
        self.agent = RandomAgent(self.env)
        self.n_episode = args.n_episodes

        self.list_of_rewards = []

    def run(self):
        for _ in range(self.n_episode):
            observation = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.agent.choose_action(observation)
                observation, reward, done, info = self.env.step(action)
                total_reward += reward

            self.list_of_rewards.append(total_reward)
        plot_rewards(self.list_of_rewards, plot=True, save=False)
        self.env.close()

