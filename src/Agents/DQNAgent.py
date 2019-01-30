import numpy as np
import tensorflow as tf

class DQNAgent:

    def __init__(self, env):

        # to make this output stable across runs
        def reset_graph(seed=42):
            tf.reset_default_graph()
            tf.set_random_seed(seed)
            np.random.seed(seed)
        reset_graph()

        self.env = env
        self.input_height = 96
        self.input_width = 80

        self.input_channels = 1
        self.conv_n_maps = [32, 64, 64]
        self.conv_kernel_sizes = [(8,8), (4,4), (3,3)]
        self.conv_strides = [4, 2, 1]
        self.conv_paddings = ["SAME"] * 3
        self.conv_activation = [tf.nn.relu] * 3
        self.n_hidden_in = 64 * 12 * 10
        self.n_hidden = 512
        self.hidden_activation = tf.nn.relu
        self.n_outputs = env.action_space.n
        self.initializer = tf.variance_scaling_initializer()

        self.learning_rate = 0.001
        self.momentum = 0.95

        self.memory = ReplayMemory()

        self.eps_min = 0.1
        self.eps_max = 1.0
        self.eps_decay_steps = 200000

        self.dueling = True

        def q_network(X_state, name):
            prev_layer = X_state / 128.0
            with tf.variable_scope(name) as scope:
                for n_maps, kernel_size, strides, padding, activation in zip(
                        self.conv_n_maps, self.conv_kernel_sizes, self.conv_strides,
                        self.conv_paddings, self.conv_activation):
                    prev_layer = tf.layers.conv2d(
                        prev_layer, filters=n_maps, kernel_size=kernel_size,
                        strides=strides, padding=padding, activation=activation,
                        kernel_initializer=self.initializer)
                last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, self.n_hidden_in])
                hidden = tf.layers.dense(last_conv_layer_flat, self.n_hidden,
                                         activation=self.hidden_activation,
                                         kernel_initializer=self.initializer)
                outputs = tf.layers.dense(hidden, self.n_outputs,
                                          kernel_initializer=self.initializer)
                if self.dueling:
                    hidden2 = tf.layers.dense(last_conv_layer_flat, self.n_hidden,
                                             activation=self.hidden_activation,
                                             kernel_initializer=self.initializer)
                    estimate_value = tf.layers.dense(hidden2, 1,
                                              kernel_initializer=self.initializer)
                    advantage = tf.subtract(outputs,tf.reduce_mean(outputs, axis=1, keepdims=True))
                    outputs = tf.add(estimate_value, advantage)

            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope=scope.name)
            trainable_vars_by_name = {var.name[len(scope.name):]: var
                                      for var in trainable_vars}
            return outputs, trainable_vars_by_name

        self.X_state = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width,
                                            self.input_channels])
        self.online_q_values, self.online_vars = q_network(self.X_state, name="q_networks/online")
        self.target_q_values, self.target_vars = q_network(self.X_state, name="q_networks/target")

        self.copy_ops = [target_var.assign(self.online_vars[var_name])
                    for var_name, target_var in self.target_vars.items()]
        self.copy_online_to_target = tf.group(*self.copy_ops)

        with tf.variable_scope("train"):
            self.X_action = tf.placeholder(tf.int32, shape=[None])
            self.y = tf.placeholder(tf.float32, shape=[None, 1])
            q_value = tf.reduce_sum(self.online_q_values * tf.one_hot(self.X_action, self.n_outputs),
                                    axis=1, keepdims=True)
            error = tf.abs(self.y - q_value)
            clipped_error = tf.clip_by_value(error, 0.0, 1.0)
            linear_error = 2 * (error - clipped_error)
            self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate, momentum=self.momentum)
            self.training_op = optimizer.minimize(self.loss, global_step=self.global_step)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()


    def preprocess_observation(self, observation):
        img = observation[5:197:2, ::2]  # downsize
        img = img.sum(axis=2)  # to greyscale
        img = (img / (3*128) - 128)
        return img.reshape(96, 80, 1)

    def remember(self, data, weight=0):
        self.memory.append(data, weight)

    def epsilon_greedy(self, q_values, step):
        epsilon = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs) # random action
        else:
            return np.argmax(q_values) # optimal action

    def sample_memories(self, batch_size):
        cols = [[], [], [], [], []] # state, action, reward, next_state, done
        for memory in self.memory.sample(batch_size):
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)


class ReplayMemory:
    def __init__(self):
        self.maxlen = 1000000
        self.buf = np.empty(shape=self.maxlen, dtype=np.object)
        self.buf_weight = np.zeros(shape=self.maxlen)
        self.index = 0
        self.length = 0

    def append(self, data, weight=0):
        self.buf[self.index] = data
        self.buf_weight[self.index] = weight
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    def sample(self, batch_size, with_replacement=True, prioritized=True):
        if prioritized:
            self.buf_weight = self.buf_weight / np.sum(self.buf_weight)
            indices = np.random.choice(range(self.length), size=min(batch_size, self.length), p=self.buf_weight[:self.length])
            return self.buf[indices]
        else:
            if with_replacement:
                indices = np.random.randint(self.length, size=batch_size) # faster
            else:
                indices = np.random.permutation(self.length)[:batch_size]
            return self.buf[indices]
