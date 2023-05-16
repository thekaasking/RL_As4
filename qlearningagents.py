import random
import numpy as np
import scipy as sp
import gymnasium as gym
from Helper import LearningCurvePlot, smooth
#
# TODO
# N_actrions fixen > dividebyzerowarning in select_action()
# Fix state indexing in select_action > error l17 IndexError: arrays used as indices must be of integer (or boolean) type


class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        if np.random.random() < 1 - self.epsilon + self.epsilon/self.n_actions:
            return np.argmax(self.Q[state,:])
        else:
            actions = [0,1,2,3]
            action = np.random.choice(actions, 1)
            maxaction = np.argmax(self.Q[state,:])
            while action==maxaction:
                action = np.random.choice(actions, 1)
            return action
        
    def update(self, state, action, reward, alpha, newstate):
        self.Q[state][action] = self.Q[state][action] + alpha * (reward+np.max(self.Q[newstate,:])-self.Q[state][action])


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        action = np.argmax(self.Q[state,:])
        if np.random.random() > self.epsilon:
            return action
        else:
            a = np.random.choice(range(self.Q[state,:].size))
            while a == action:
                a = np.random.choice(range(self.Q[state,:].size))
            action = a
        return action
        
    def update(self, state, action, reward, alpha, state_1, action_1):
        self.Q[state][action] = self.Q[state][action] + alpha*(reward + (self.Q[state_1][action_1])-self.Q[state][action])


def cur_state(env):
    # do ando an action])
    # empty step
    new_state, b, c, d, e = env.step(0)
    return new_state


def run_repetitions(n_episodes, alpha, epsilon):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    rewards = np.zeros(n_episodes)
    done = False
    dim_x = len(env.observation_space.low)
    dim_y = len(env.observation_space.high)
    state_size = dim_x*dim_y
    y = np.zeros(state_size)
    
    n_actions = env.action_space.sample()
    #print(state_size)
    #print(n_actions)
    pi = QLearningAgent(n_actions=n_actions, epsilon = epsilon, n_states = state_size)
    for i in range(n_episodes):
        env.reset()
        while not done:
            state = env.state
            a = pi.select_action(state) # select action
            new_state, reward, done = env.step(a) # sample reward
            y[state] = np.argmax(pi.Q[state,:])
           # pi.update(state, a, reward, alpha, cur_state(env)) # update policy
            pi.update(state, a, reward, alpha, new_state) # update policy
            rewards[i] += reward
    return y, rewards


def run_repetitions_SARSA(n_episodes, alpha, epsilon):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    rewards = np.zeros(n_episodes)
    
    dim_x = len(env.observation_space.low)
    dim_y = len(env.observation_space.high)

    y = np.zeros(dim_x)
    
    n_actions = env.action_size.sample

    pi = SARSAAgent(n_actions=n_actions, epsilon = epsilon, n_states = env.state_size())
    for i in range(n_episodes):
        env.reset()
        a = pi.select_action(env.state()) # select action
        while not env.done():
            state = env.state()
            r = env.step(a) # sample reward
            action_1 = pi.select_action(env.state()) # select next action
            y[state] = np.argmax(pi.Q[state,:])
            pi.update(state, a, r, alpha, env.state(), action_1) # update policy
            rewards[i] += r
            a = action_1
    return y, rewards


def experiment(n_episodes, n_repetitions, smoothing_window, epsilon, alpha_values):

    # Assignment 1: Q-learning ---------------------------------------------------
    LC_Qlearning = LearningCurvePlot(title = "Learning curve for Q-learning")
    for alpha in alpha_values:
        results = np.zeros((n_repetitions, n_episodes))
        for i in range(n_repetitions):
            temp, results[i]  = run_repetitions(n_episodes, alpha, epsilon)
        avgs = np.sum(results, axis = 0)/n_repetitions
        LC_Qlearning.add_curve(smooth(avgs,smoothing_window),label=f'alpha={alpha}, smoothed')
    LC_Qlearning.save(name='learning_curve_Q_learning.png')

    # Assignment 2: SARSA ---------------------------------------------------
    # LC_SARSA = LearningCurvePlot(title = "Learning curve for SARSA")
    # for alpha in alpha_values:
    #     results = np.zeros((n_repetitions, n_episodes))
    #     for i in range(n_repetitions):
    #         temp, results[i]  = run_repetitions_SARSA(n_episodes, alpha, epsilon)
    #     avgs = np.sum(results, axis = 0)/n_repetitions
    #     LC_SARSA.add_curve(smooth(avgs,smoothing_window),label=f'alpha={alpha}, smoothed')
    # LC_SARSA.save(name='learning_curve_SARSA.png')


if __name__ == '__main__':
    
    # experiment settings
    smoothing_window = 31
    n_episodes = 1000
    n_rep = 100
    epsilon = 0.1
    alpha = 0.1
    alpha_values = [0.01, 0.1, 0.5, 0.9]

    experiment(n_episodes, n_rep, smoothing_window, epsilon, alpha_values)

    np.set_printoptions(threshold=np.inf)
    
   # array_y, rewards = run_repetitions(n_episodes, alpha, epsilon)
    
   # array_y, rewards = run_repetitions_SARSA(n_episodes, alpha, epsilon)