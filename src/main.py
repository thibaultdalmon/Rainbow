from ReinforcementLearning.Trainer import Trainer

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--env_name', type=str, default='Seaquest-v0')
parser.add_argument('--n_episodes', type=int, default=4000000)

args = parser.parse_args()

trainer = Trainer(args)
trainer.run_dqn()
