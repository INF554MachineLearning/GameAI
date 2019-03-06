import tensorflow as tf
from collections import deque
import numpy as np


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            # Input is 16x13x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=8,
                                          kernel_size=[5, 5],
                                          strides=[1, 1],
                                          padding="VALID",
                                          name="conv1")

            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=16,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding="VALID",
                                          name="conv2")

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=32,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding="VALID",
                                          name="conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.layers.flatten(self.conv3_out, name="flatten")

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=16,
                                      activation=tf.nn.elu,
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc,
                                          units=self.action_size,
                                          activation=None)

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
