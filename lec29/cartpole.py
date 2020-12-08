
import os 
## Suppress TensorFlow Info and Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from os.path import exists

import gym
import random
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout

##from keras.optimizers import rmsprop, Adam ## Older versions used rmsprop
from keras.optimizers import RMSprop, Adam

## With TF2 we need this or otherwise it will be too slow
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from math import exp,cos
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import seed,randn

from collections import deque
from statistics import mean
import h5py

LEARNING_RATE = 1e-3
MAX_MEMORY = 100000
BATCH_SIZE = 32
GAMMA = 0.975
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001
EPISODES = 10000

class Network:

    def __init__(self, observation_space, action_space):

        self.action_space = action_space
        self.memory = deque(maxlen=MAX_MEMORY)
        self.exploration_rate = 1.0

        self.model = Sequential()
        self.model.add(Dense(32, input_shape=(observation_space,), activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.action_space, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def take_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(0, self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        else:
            minibatch = random.sample(self.memory, BATCH_SIZE)

            ## TODO:
            ## This loop trains one sample at a time the model, but we could
            ## use the whole minibatch at once
        
            for state, action, reward, state_next, done in minibatch:
                Q = reward
                if not done:
                    Q = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
                Q_values = self.model.predict(state)
                Q_values[0][action] = Q
                self.model.fit(state, Q_values, verbose=0)

                
            self.exploration_rate *= EXPLORATION_DECAY
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def get_model(self):
        return self.model

    def load_model(self,model_name):
        self.model = load_model(model_name)
    


class TrainSolver:

    def __init__(self, max_episodes):
        self.max_episodes = max_episodes
        self.score_table = deque(maxlen=400)
        self.average_of_last_runs = None
        self.model = None
        self.play_episodes = 100
        env = gym.make('CartPole-v1')
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        self.solver = Network(observation_space, action_space)

    def train(self):
        
        env = gym.make('CartPole-v1')
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n

        print("---------------------------------")
        print("Solver starts")
        print("---------------------------------")

        self.model = self.solver.get_model()
           
        episode = 0
        while episode < self.max_episodes:

            episode += 1
            state = env.reset()

            ## Hack a more diverse initial random position
            x, x_dot, theta, theta_dot = env.state
            x = randn()*3;
            env.state = (x,x_dot,theta,theta_dot)

            state = np.reshape(np.array(env.state), [1, observation_space])
            
            step = 0
            while True:

                env.render()
                
                step += 1
                action = self.solver.take_action(state)
                state_next, reward, done, info = env.step(action)

                state_next = np.reshape(state_next, [1, observation_space])

                ## State is a vector with one observation
                ## Type: Box(4)
                ## Num  Observation                 Min         Max
                ## 0    Cart Position             -4.8            4.8
                ## 1    Cart Velocity             -Inf            Inf
                ## 2    Pole Angle                 -24 deg        24 deg
                ## 3    Pole Velocity At Tip      -Inf            Inf
                
                ## Prefer to be in the middle and vertical
                reward = exp(-0.5*abs((state_next[0][0]**2)/0.5)) * \
                         cos(state_next[0][2])
                
                ##if not done:
                ##    reward = reward
                ##else:
                ##    reward = exp(-0.5*abs((state_next[0][0]**2)/0.5)) - 0.2

                self.solver.add_to_memory(state, action, reward, state_next, done)
                state = state_next

                print("  State: " + str(state) +
                      ", reward: " + str(reward) +
                      "                 ",
                      end='\r', flush=True)
                
                if done:
                    print("\nRun: " + str(episode) +
                          ", exploration: "+str(self.solver.exploration_rate) +
                          ", score: " + str(step) +
                          ", mem: " + str(len(self.solver.memory)))

                    break

                ## Train the network
                self.solver.experience_replay()


    def return_trained_model(self):
        return self.model

    def save_model(self):
        self.model.save('cartpole_model.h5')

    def load_model(self):
        filename = 'cartpole_model.h5'
        if os.path.exists(filename):
            self.solver.load_model(filename)
            self.model = self.solver.get_model()
        else:
            print("File '" + filename + "' does not exist. Ignoring")

        

RL=TrainSolver(EPISODES)
RL.load_model();
RL.train()
RL.save_model()
