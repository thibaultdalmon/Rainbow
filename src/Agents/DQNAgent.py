import numpy as np

from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model

class DQNAgent:

    def __init__(self, env):
        self.env = env
        self.action_size = self.action_space.n
        self.input_height = pass
        self.input_width = pass
        self.discount_rate = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.learning_rate = 0.001
        self.model = self.qNetwork()
        self.memory = ReplayMemory()

    def choose_action(self, observation):
        if np.random.rand() <= self.epsilon:
            return self.action_size.sample()
        processed_obs = self.preprocessing(observation)
        act_values = self.model.predict(processed_obs)
        return np.argmax(act_values[0])

    def preprocessing(self, observation):
        pass

    def remember(self, data):
        self.memory.append(data)

    def qNetwork(self):
        input = Input(shape=(self.input_height, self.width))
        conv1 = Conv2D(filters=32, kernel_size=(8,8), activation='relu')(input)
        conv2 = Conv2D(filters=32, kernel_size=(4,4), activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv2)
        flat = Flatten()(conv3)
        dense1 = Dense(512, activation='relu')(flat)
        dense2 = Dense(self.action_size)(dense1)
        model = Model(inputs=input, outputs=predictions)
        model.compile(loss = "mse",
            optimizer = Adam(lr=self.learning_rate),
            metrics=["accuracy"])
        return model

    def train(self, batch_size):
        states, actions, rewards, next_states, done = (
            self.memory.sample_memories(batch_size))
        targets = rewards
        q_values = self.model.predict(next_states)
        max_q_values = np.max(q_values, axis=1, keepdims=True)
        y_values = rewards + done * discount_rate * max_q_values
        self.model.fit(states, y_values, epochs=1, verbose=0)


class ReplayMemory:
    def __init__(self, maxlen):
        self.maxlen = 500000
        self.buf = np.empty(shape=maxlen, dtype=np.object)
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
        cols = [[], [], [], [], []] # state, action, reward, next_state, continue
        for memory in self.sample(batch_size):
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)
