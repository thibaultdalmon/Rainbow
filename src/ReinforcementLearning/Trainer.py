from Agents.RandomAgent import RandomAgent
from Agents.DQNAgent import DQNAgent
from utils.Plots import RewardPlot

import gym
import os
import numpy as np
import tensorflow as tf

class Trainer:

    def __init__(self, args):

        # Using Double Qlearning (see Rainbow paper)
        self.doubleQlearning = True

        # Using multi-step RL
        self.n_step = 10

        self.env = gym.make(args.env_name)
        self.agent = RandomAgent(self.env)
        self.agent_dqn = DQNAgent(self.env)

        self.n_episode = args.n_episodes # nb of games to play
        self.training_start = 10000  # start training after 10,000 game iterations
        self.training_interval = 4  # run a training step every 4 game iterations
        self.save_steps = 1000  # save the model every 1,000 training steps
        self.copy_steps = 1000  # copy online DQN to target DQN every 10,000 training steps
        self.discount_rate = 0.95
        self.skip_start = 0  # Skip the start of every game
        self.batch_size = 32
        self.iteration = 0  # game iterations
        self.step = 0   # global training step
        self.loss_val = np.infty
        self.game_length = 0
        self.total_max_q = 0
        self.mean_max_q = 0.0
        self.total_reward = 0
        self.mean_reward = 0.0
        self.checkpoint_path = "../DQN/DQN_test.ckpt"
        self.w = 0.5

        # plotting data
        self.reward_plot = RewardPlot()

    # running random agent
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

    # running Deep Qlearning agent
    def run_dqn(self):
        done = True
        list_of_rewards = []
        list_of_states_per_episode = []
        list_of_actions_per_episode = []
        list_of_rewards_per_episode = []
        list_of_done_per_episode = []
        list_of_next_state_per_episode = []
        list_of_weights = []
        idx_to_store = []
        with tf.Session() as sess:

            # Restoring model if previously trained
            if os.path.isfile(self.checkpoint_path + ".index"):
                self.agent_dqn.saver.restore(sess, self.checkpoint_path)
            else:
                self.agent_dqn.init.run()
                self.agent_dqn.copy_online_to_target.run()

            # Running games and training
            while True:

                # Initializing the number of training steps
                step = self.agent_dqn.global_step.eval()
                if step >= self.n_episode:
                    break
                self.iteration += 1

                # Printing statistics
                print("\rIteration {}\tTraining step {}/{} ({:.1f})%\tLoss {:5f}\tMean Max-Q {:5f}   ".format(
                    self.iteration, step, self.n_episode, step * 100 / self.n_episode,
                    self.loss_val, self.mean_max_q), end="")

                # Checking if previous game is over
                if done:  # game over, start again
                    obs = self.env.reset()
                    for skip in range(self.skip_start): # skip the start of each game
                        obs, reward, done, info = self.env.step(0)
                    state = self.agent_dqn.preprocess_observation(obs)
                    list_of_states_per_episode = []
                    list_of_actions_per_episode = []
                    list_of_rewards_per_episode = []
                    list_of_done_per_episode = []
                    list_of_next_state_per_episode = []
                    idx_to_store = []
                    list_of_weights = []

                # Sampling noisy variables
                # If Noisy Nets are not in the Qnetwork, does nothing
                self.agent_dqn.reset_network()

                # Online DQN evaluates what to do
                full_dict = self.agent_dqn.epsilon
                full_dict[self.agent_dqn.X_state] = [state]
                q_values = self.agent_dqn.online_q_values.eval(feed_dict=full_dict)
                action = self.agent_dqn.epsilon_greedy(q_values, step)
                list_of_states_per_episode.append(state)
                list_of_actions_per_episode.append(action)
                idx_to_store.append(self.game_length)

                # Online DQN plays
                obs, reward, done, info = self.env.step(action)
                next_state = self.agent_dqn.preprocess_observation(obs)
                list_of_rewards_per_episode.append(reward)
                list_of_done_per_episode.append(done)
                list_of_next_state_per_episode.append(next_state)

                # Calculation of the priority for replay
                full_dict = self.agent_dqn.epsilon
                full_dict[self.agent_dqn.X_state] = [state]
                q_bar_values = self.agent_dqn.target_q_values.eval(feed_dict=full_dict)
                list_of_weights.append(np.power(np.abs(reward + self.discount_rate * np.max(q_bar_values) - q_values[0, action]), self.w))

                # Let's memorize what happened
                if len(list_of_rewards_per_episode) >= self.n_step:
                    reward_to_store = np.sum([self.discount_rate ** i_reward * list_of_rewards_per_episode[k_reward]
                                              for i_reward, k_reward in enumerate(range(- self.n_step, 0))])
                    to_store = (list_of_states_per_episode[- self.n_step],
                                list_of_actions_per_episode[- self.n_step],
                                reward_to_store,
                                list_of_next_state_per_episode[- self.n_step],
                                1.0 - list_of_done_per_episode[- self.n_step],)
                    self.agent_dqn.memory.append(to_store, list_of_weights[- self.n_step])
                    idx_to_store.pop(- self.n_step)
                state = next_state

                # Compute statistics for tracking progress
                self.total_max_q += q_values.max()
                self.total_reward += reward
                self.game_length += 1
                if done:
                    # Let's memorize the end of the episode
                    for idx in idx_to_store:
                        reward_to_store = np.sum([self.discount_rate ** i_reward * list_of_rewards_per_episode[k_reward]
                                              for i_reward, k_reward in enumerate(range(idx, self.game_length))])
                        to_store = (list_of_states_per_episode[idx],
                                list_of_actions_per_episode[idx],
                                reward_to_store,
                                list_of_next_state_per_episode[idx],
                                1.0 - list_of_done_per_episode[idx])
                        self.agent_dqn.memory.append(to_store, list_of_weights[idx])

                    # Statistics
                    self.mean_max_q = self.total_max_q / self.game_length
                    self.mean_reward = self.total_reward / self.game_length
                    list_of_rewards.append(self.total_reward)
                    self.reward_plot.update_and_plot(list_of_rewards, plot=True, save=False)
                    self.total_max_q = 0.0
                    self.total_reward = 0.0
                    self.game_length = 0

                if self.iteration < self.training_start or self.iteration % self.training_interval != 0:
                    continue # only train after warmup period and at regular intervals

                # Sample memories and use the target DQN to produce the target Q-Value
                self.agent_dqn.reset_network()
                X_state_val, X_action_val, rewards, X_next_state_val, continues = (
                    self.agent_dqn.sample_memories(self.batch_size))

                # Compute R_t+1 + gamma.Q(S_t+1, argmax(...)) in Double Q-learning
                if self.doubleQlearning:
                    next_online_actions = self.agent_dqn.online_q_values
                    max_action = tf.argmax(next_online_actions, axis=1)
                    q_values = self.agent_dqn.target_q_values
                    next_q_values = tf.reduce_sum(
                        q_values * tf.one_hot(max_action, self.env.action_space.n),
                        axis=1, keepdims=True)
                    full_dict = self.agent_dqn.epsilon
                    full_dict[self.agent_dqn.X_state] = X_next_state_val
                    target_q_values = next_q_values.eval(
                        feed_dict=full_dict)
                    y_val = rewards + continues * self.discount_rate * target_q_values

                # Compute R_t+1 + gamma.max(Q(S_t+1, ...)) in Q-learning
                else:
                    full_dict = self.agent_dqn.epsilon
                    full_dict[self.agent_dqn.X_state] = X_next_state_val
                    next_q_values = self.agent_dqn.target_q_values.eval(
                        feed_dict=full_dict)
                    max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
                    y_val = rewards + continues * self.discount_rate * max_next_q_values

                # Train the online DQN
                full_dict = self.agent_dqn.epsilon
                full_dict[self.agent_dqn.X_state] = X_state_val
                full_dict[self.agent_dqn.X_action] = X_action_val
                full_dict[self.agent_dqn.y] = y_val
                _, self.loss_val = sess.run([self.agent_dqn.training_op, self.agent_dqn.loss],
                    feed_dict=full_dict)

                # Regularly copy the online DQN to the target DQN
                if step % self.copy_steps == 0:
                    self.agent_dqn.copy_online_to_target.run()

                # And save regularly
                if step % self.save_steps == 0:
                    self.agent_dqn.saver.save(sess, self.checkpoint_path)
