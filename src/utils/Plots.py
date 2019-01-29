from matplotlib import pyplot as plt
import numpy as np


class RewardPlot:

    def __init__(self):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        self.line_reward = plt.Line2D([], [], label='rewards', color='b')
        self.line_mean = plt.Line2D([], [], label='average reward', color='r')
        self.ax.add_line(self.line_reward)
        self.ax.add_line(self.line_mean)
        self.list_of_rewards = np.array([])

    def update_and_plot(self, list_of_rewards, plot=True, save=False, filename=None):
        self.list_of_rewards = np.array(list_of_rewards)
        max_reward = np.max(list_of_rewards)
        min_reward = np.min(list_of_rewards)
        mean_reward = np.mean(list_of_rewards)
        n_data = self.list_of_rewards.size

        self.ax.set_xlim(left=0, right=max(1, n_data - 1))
        self.ax.set_ylim(bottom=min(0, min_reward), top=max(1, max_reward))

        self.line_reward.set_xdata(range(n_data))
        self.line_reward.set_ydata(self.list_of_rewards)

        self.line_mean.set_xdata(range(n_data))
        self.line_mean.set_ydata([mean_reward for _ in range(n_data)])
        self.ax.legend()

        if plot:
            plt.draw()
            plt.pause(0.0000001)

        if save:
            self.fig.savefig(filename, format='png')
