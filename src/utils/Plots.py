from matplotlib import pyplot as plt
import numpy as np


def plot_rewards(list_of_rewards, plot=True, save=False, filename=None):
    max_reward = max(list_of_rewards)
    min_reward = min(list_of_rewards)
    n_data = len(list_of_rewards)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlim(left=0, right=n_data - 1)
    ax.set_ylim(bottom=min(0, min_reward), top=max(0, max_reward))
    ax.plot(list_of_rewards)

    if plot:
        fig.show()

    if save:
        fig.savefig(filename, format='png')
