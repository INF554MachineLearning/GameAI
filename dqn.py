import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque

import random

import gym, ppaquette_gym_super_mario
import sys
import model


# stack_size = 4  # We stack 4 frames
#
# height = 13
# width = 16

# Initialize deque with zero-images one array for each image
# stacked_frames = deque([np.zeros((height, width), dtype=np.int) for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, frame):
    # Preprocess frame
    # frame = preprocess_frame(state)
    # frame = np.expand_dims(state, axis=-1)

    # Append frame to deque, automatically removes the oldest frame
    stacked_frames.append(frame)

    # Build the stacked state (first dimension specifies different frames)
    stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


class AgentDQN:

    def __init__(self, possible_actions, lr, tot_episodes, max_steps, batch_size, explore_start,
                 explore_stop, decay_rate, gamma, memory_size, env, stack_size):
        self.action_size = len(possible_actions)
        self.learning_rate = lr
        self.total_episodes = tot_episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        # Exploration parameters for epsilon greedy strategy
        self.explore_start = explore_start  # exploration probability at start
        self.explore_stop = explore_stop  # minimum exploration probability
        self.decay_rate = decay_rate  # exponential decay rate for exploration prob
        self.gamma = gamma  # Discounting rate
        self.pretrain_length = self.batch_size
        self.memory_size = memory_size

        self.possible_actions = possible_actions
        self.actions_for_nn = np.eye(self.action_size)

        self.env = env
        height, width = self.env.observation_space.shape
        self.state_size = (height, width, stack_size)

        self.memory = model.Memory(max_size=self.memory_size)

        self.DQNetwork = model.DQNetwork(self.state_size, self.action_size, self.learning_rate)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def init_memory(self):
        # Instantiate memory
        print("Initialize memory !")
        for i in range(self.pretrain_length):
            # If it's the first step
            if i == 0:

                frame = self.env.reset()
                stacked_frames = deque([np.zeros_like(frame, dtype=np.int) for i in range(self.state_size[-1])],
                                       maxlen=4)
                stacked_frames.append(frame)
                stacked_frames.append(frame)
                stacked_frames.append(frame)
                stacked_frames.append(frame)
                # Stack the frames
                state = np.stack(stacked_frames, axis=2)

            # Get the next_state, the rewards, done by taking a random action
            choice = random.randint(1, len(self.possible_actions)) - 1
            action = self.possible_actions[choice]
            ac_for_nn = self.actions_for_nn[choice]
            next_frame, reward, done, info = self.env.step(action)

            # Stack the frames
            next_state, stacked_frames = stack_frames(stacked_frames, next_frame)

            # If the episode is finished (we're dead 3x)
            if done:
                # We finished the episode
                next_state = np.zeros(state.shape)

                # Add experience to memory
                self.memory.add((state, ac_for_nn, reward, next_state, done))

                # Start a new episode
                frame = self.env.reset()

                # Stack the frames
                stacked_frames = deque([np.zeros_like(frame, dtype=np.int) for i in range(self.state_size[-1])],
                                       maxlen=4)
                stacked_frames.append(frame)
                stacked_frames.append(frame)
                stacked_frames.append(frame)
                stacked_frames.append(frame)
                # Stack the frames
                state = np.stack(stacked_frames, axis=2)

            else:
                # Add experience to memory
                self.memory.add((state, ac_for_nn, reward, next_state, done))

                # Our new state is now the next_state
                state = next_state

    """
    This function will do the part
    With ϵϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
    """

    def predict_action(self, decay_step, state):
        # EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        # First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(
            -self.decay_rate * decay_step)

        if explore_probability > exp_exp_tradeoff:
            # Make a random action (exploration)
            choice = random.randint(1, len(self.possible_actions)) - 1
            action = self.possible_actions[choice]

        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = self.sess.run(self.DQNetwork.output,
                               feed_dict={self.DQNetwork.inputs_: state.reshape((1, *state.shape))})

            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = self.possible_actions[choice]

        return self.actions_for_nn[choice], action, explore_probability

    def train(self):
        print("Begin Training ..")
        rewards_list = []
        # Initialize the variables
        self.sess.run(tf.global_variables_initializer())
        # saver.restore(sess, "./model.ckpt")

        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        for episode in range(self.total_episodes):
            init_time = time.time()
            # Set step to 0
            step = 0

            # Initialize the rewards of the episode
            episode_rewards = []

            # Make a new episode and observe the first state
            frame = self.env.reset()
            stacked_frames = deque([np.zeros_like(frame, dtype=np.int) for i in range(self.state_size[-1])], maxlen=4)
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            # Stack the frames
            state = np.stack(stacked_frames, axis=2)

            # Remember that stack frame function also call our preprocess function.
            # state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step < self.max_steps:
                step += 1

                # Increase decay_step
                decay_step += 1

                # Predict the action to take and take it
                ac_for_nn, action, explore_probability = self.predict_action(decay_step, state)

                # Perform the action and get the next_state, reward, and done information
                next_frame, reward, done, info = self.env.step(action)

                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if done:
                    # The episode ends so no next state
                    next_frame = np.zeros((self.state_size[0], self.state_size[1]), dtype=np.int)

                    next_state, stacked_frames = stack_frames(stacked_frames, next_frame)

                    # Set step = max_steps to end the episode
                    step = self.max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)
                    finish_time = time.time()

                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Explore P: {:.4f}'.format(explore_probability),
                          'Training Loss {:.4f}'.format(loss),
                          'Duration {:.4f}'.format(finish_time - init_time))

                    rewards_list.append((episode, total_reward))

                    # Store transition <st,at,rt+1,st+1> in memory D
                    self.memory.add((state, ac_for_nn, reward, next_state, done))

                else:
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_frame)

                    # Add experience to memory
                    self.memory.add((state, ac_for_nn, reward, next_state, done))

                    # st+1 is now our current state
                    state = next_state

                # LEARNING PART
                # Obtain random mini-batch from memory
                batch = self.memory.sample(self.batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                # Get Q values for next_state
                Qs_next_state = self.sess.run(self.DQNetwork.output, feed_dict={self.DQNetwork.inputs_: next_states_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        target = rewards_mb[i] + self.gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = self.sess.run([self.DQNetwork.loss, self.DQNetwork.optimizer],
                                        feed_dict={self.DQNetwork.inputs_: states_mb,
                                                   self.DQNetwork.target_Q: targets_mb,
                                                   self.DQNetwork.actions_: actions_mb})

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = self.saver.save(self.sess, "./model.ckpt")
                print("Model Saved")


#
# if training == True:
#     with tf.Session() as sess:
#


def main():
    level = "1-1"
    env = gym.make('ppaquette/SuperMarioBros-' + level + '-Tiles-v0')

    possible_actions = [[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 1, 1, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0]]

    learning_rate = 0.00025  # Alpha (aka learning rate)
    stack_size = 4
    # TRAINING HYPERPARAMETERS
    total_episodes = 100  # Total episodes for training
    max_steps = 5000  # Max possible steps in an episode
    batch_size = 64  # Batch size

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0  # exploration probability at start
    explore_stop = 0.01  # minimum exploration probability
    decay_rate = 0.00001  # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.9  # Discounting rate

    # MEMORY HYPERPARAMETERS
    memory_size = 1000000  # Number of experiences the Memory can keep

    mario_agent = AgentDQN(possible_actions, learning_rate, total_episodes, max_steps, batch_size, explore_start,
                 explore_stop, decay_rate, gamma, memory_size, env, stack_size)

    mario_agent.init_memory()

    mario_agent.train()


if __name__ == '__main__':
    main()
