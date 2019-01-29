from Agents.RandomAgent import RandomAgent
from Agents.DQNAgent import DQNAgent
from utils.Plots import RewardPlot

import gym
import os
import numpy as np
import tensorflow as tf

class Trainer:

    def __init__(self, args):
        self.env = gym.make(args.env_name)
        self.agent = RandomAgent(self.env)
        self.agent_dqn = DQNAgent(self.env)
        self.n_episode = args.n_episodes # nb of games to play
        self.training_start = 10000  # start training after 10,000 game iterations
        self.training_interval = 4  # run a training step every 4 game iterations
        self.save_steps = 1000  # save the model every 1,000 training steps
        self.copy_steps = 10000  # copy online DQN to target DQN every 10,000 training steps
        self.discount_rate = 0.99
        self.skip_start = 0  # Skip the start of every game
        self.batch_size = 50
        self.iteration = 0  # game iterations
        self.step = 0
        self.loss_val = np.infty
        self.game_length = 0
        self.total_max_q = 0
        self.mean_max_q = 0.0
        self.checkpoint_path = "../DQN/DQN_test.ckpt"

        self.reward_plot = RewardPlot()

    def run(self):
        list_of_rewards = []
        for epoch in range(self.n_episode):
            state = self.env.reset()
            state = self.agent_dqn.preprocessing(state)
            done = False
            total_reward = 0
            print(epoch)

            while not done:
                action = self.agent_dqn.choose_action(state)
                next_state, reward, done, info = self.env.step(action)

                next_state = self.agent_dqn.preprocessing(next_state)
                self.agent_dqn.remember((state, action, reward, next_state, done))

                state = next_state

                total_reward += reward

            if epoch % 10 == 9:
                self.agent_dqn.train(50)

            self.step += 1
            list_of_rewards.append(total_reward)
            self.reward_plot.update_and_plot(list_of_rewards, plot=True, save=False)
        self.env.close()

    def run_dqn(self):
        done = True
        with tf.Session() as sess:
            if os.path.isfile(self.checkpoint_path + ".index"):
                self.agent_dqn.saver.restore(sess, self.checkpoint_path)
            else:
                self.agent_dqn.init.run()
                self.agent_dqn.copy_online_to_target.run()
            while True:
                step = self.agent_dqn.global_step.eval()
                if step >= self.n_episode:
                    break
                self.iteration += 1
                print("\rIteration {}\tTraining step {}/{} ({:.1f})%\tLoss {:5f}\tMean Max-Q {:5f}   ".format(
                    self.iteration, step, self.n_episode, step * 100 / self.n_episode,
                    self.loss_val, self.mean_max_q), end="")
                if done: # game over, start again
                    obs = self.env.reset()
                    for skip in range(self.skip_start): # skip the start of each game
                        obs, reward, done, info = self.env.step(0)
                    state = self.agent_dqn.preprocess_observation(obs)

                # Online DQN evaluates what to do
                q_values = self.agent_dqn.online_q_values.eval(feed_dict={self.agent_dqn.X_state: [state]})
                action = self.agent_dqn.epsilon_greedy(q_values, step)

                # Online DQN plays
                obs, reward, done, info = self.env.step(action)
                next_state = self.agent_dqn.preprocess_observation(obs)

                # Let's memorize what happened
                self.agent_dqn.memory.append((state, action, reward, next_state, 1.0 - done))
                state = next_state

                # Compute statistics for tracking progress
                self.total_max_q += q_values.max()
                self.game_length += 1
                if done:
                    self.mean_max_q = self.total_max_q / self.game_length
                    self.total_max_q = 0.0
                    self.game_length = 0

                if self.iteration < self.training_start or self.iteration % self.training_interval != 0:
                    continue # only train after warmup period and at regular intervals

                # Sample memories and use the target DQN to produce the target Q-Value
                X_state_val, X_action_val, rewards, X_next_state_val, continues = (
                    self.agent_dqn.sample_memories(self.batch_size))
                next_q_values = self.agent_dqn.target_q_values.eval(
                    feed_dict={self.agent_dqn.X_state: X_next_state_val})
                max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
                y_val = rewards + continues * self.discount_rate * max_next_q_values

                # Train the online DQN
                _, self.loss_val = sess.run([self.agent_dqn.training_op, self.agent_dqn.loss],
                    feed_dict={self.agent_dqn.X_state: X_state_val,
                    self.agent_dqn.X_action: X_action_val,
                    self.agent_dqn.y: y_val})

                # Regularly copy the online DQN to the target DQN
                if step % self.copy_steps == 0:
                    self.agent_dqn.copy_online_to_target.run()

                # And save regularly
                if step % self.save_steps == 0:
                    self.agent_dqn.saver.save(sess, self.checkpoint_path)
