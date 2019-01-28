from Agents.RandomAgent import RandomAgent

import gym


class Trainer:

    def __init__(self, args):
        self.env = gym.make(args.env_name)
        self.agent = RandomAgent(self.env)
        self.n_episode = args.n_episodes

    def run(self):
        observation = self.env.reset()
        rewards = 0
        for _ in range(self.n_episode):
            self.env.render(mode='human')
            action = self.agent.choose_action(observation)
            observation, reward, done, info = self.env.step(action)
            rewards += 0
            if done:
                break
        self.env.close()

