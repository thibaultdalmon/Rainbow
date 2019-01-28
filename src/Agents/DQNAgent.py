import numpy as np

from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

class DQNAgent:

    def __init__(self, env):
        self.env = env
        self.action_size = self.env.action_space.n
        self.input_height = 105
        self.input_width = 80
        self.discount_rate = 0.95    # discount rate
        self.epsilon = 0.01  # exploration rate
        self.learning_rate = 0.001
        self.model = self.qNetwork()
        self.memory = ReplayMemory()
        self.max_q_value = 0
        self.loss_val = 0

    def choose_action(self, observation):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        act_values = self.model.predict(observation)
        self.max_q_value = np.max(act_values)
        return np.argmax(act_values[0])

    def preprocessing(self, observation):
        img = observation[::2,::2] # downsize
        img = img.sum(axis=2) # to greyscale
        img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
        return img.reshape(1, 105, 80, 1)

    def remember(self, data):
        self.memory.append(data)

    def qNetwork(self):
        input = Input(shape=(self.input_height, self.input_width,1))
        conv1 = Conv2D(filters=32, kernel_size=(8,8), activation='relu')(input)
        conv2 = Conv2D(filters=32, kernel_size=(4,4), activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv2)
        flat = Flatten()(conv3)
        dense1 = Dense(512, activation='relu')(flat)
        dense2 = Dense(self.action_size)(dense1)
        model = Model(inputs=input, outputs=dense2)

        def _loss_dqn(y_true, y_pred):
            q_value = K.sum(y_pred * K.one_hot(K.argmax(y_pred),
                                    self.action_size),
                                    axis=1, keepdims=True)
            error = K.abs(y_true - q_value)
            clipped_error = K.clip(error, 0.0, 1.0)
            linear_error = 2 * (error - clipped_error)
            loss = K.mean(K.square(clipped_error) + linear_error)
            self.loss_val = loss
            return loss

        model.compile(loss = _loss_dqn,
            optimizer = Adam(lr=self.learning_rate))
        return model

    def train(self, batch_size):
        states, actions, rewards, next_states, done = (
            self.memory.sample_memories(batch_size))
        targets = rewards
        q_values = self.model.predict(next_states)
        max_q_values = np.max(q_values, axis=1, keepdims=True)
        y_values = rewards + done * self.discount_rate * max_q_values

        self.model.fit(states, y_values, epochs=1)


class ReplayMemory:
    def __init__(self):
        self.maxlen = 500000
        self.buf = np.empty(shape=self.maxlen, dtype=np.object)
        self.index = 0
        self.length = 0

    def append(self, data):
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    def sample(self, batch_size, with_replacement=True):
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size) # faster
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]

    def sample_memories(self, batch_size):
        cols = [[], [], [], [], []] # state, action, reward, next_state, done
        for memory in self.sample(batch_size):
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return (cols[0].reshape(batch_size, 105, 80, 1), cols[1], cols[2],
                cols[3].reshape(batch_size, 105, 80, 1), cols[4])
