from ReinforcementLearning.Trainer import Trainer

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

plt.ion()

parser = argparse.ArgumentParser()

parser.add_argument('--env_name', type=str, default='Atlantis-v0')
parser.add_argument('--n_episodes', type=int, default=4000000)

args = parser.parse_args()

trainer = Trainer(args)
trainer.run_dqn()

plt.ioff()
