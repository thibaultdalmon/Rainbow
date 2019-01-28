class RandomAgent:

    def __init__(self, env):
        self.env = env

    def choose_action(self, observation):
        return self.env.action_space.sample()
