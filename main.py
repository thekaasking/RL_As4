"""
Introduction to Reinforcement Learning
            Assignment 4

Code for the RL 

By: Liva van der Velden & Razo van Berkel
Leiden University, May 2023
"""

import torch  
import gymnasium as gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from Helper import LearningCurvePlot, smooth

class PolicyNetwork(nn.Module):
    """
    Policy network that represents the agent's policy.
    """
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        """
        Initialize the PolicyNetwork class.

        :param num_inputs: Number of input features.
        :param num_actions: Number of possible actions.
        :param hidden_size: Size of the hidden layer.
        :param learning_rate: Learning rate for the optimizer.
        """
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        """
        Forward pass of the policy network.

        :param state: Input state.
        :return: Action probabilities.
        """
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x 
    
    def get_action(self, state):
        """
        Get an action based on the given state.

        :param state: Input state.
        :return: Action and corresponding log probability.
        """
        if isinstance(state, tuple):
            state = torch.from_numpy(state[0]).float().unsqueeze(0)
        else: 
            state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob
    
def update_policy(policy_network, rewards, log_probs, gamma):
    """
    Update the policy network based on the rewards and log probabilities.

    :param policy_network: PolicyNetwork instance.
    :param rewards: List of rewards.
    :param log_probs: List of log probabilities.
    :param gamma: the discount factor Gamma.
    """
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + gamma**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)
        
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    
    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()



def experiment(num_episodes, max_steps, gamma, learning_rate):
    """
    Main function to train the policy network using the REINFORCE algorithm.
    Returns the lists full of data about the run, instead of creating the figure here.
    The lists are used to create the plots in the main function.

    :param num_episodes: episodes our agent runs
    :param max_steps: maximum steps our agent does
    :return avg_numsteps: list of the average number of steps the agent took 
    :return numsteps: list of all aggregated steps
    :return all_rewards: list of all aggregated rewards
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, 128, learning_rate=learning_rate)
    

    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []

        for steps in range(max_steps):
            env.render()
            action, log_prob = policy_net.get_action(state)
            new_state, reward, done, truncated, info = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                update_policy(policy_net, rewards, log_probs, gamma)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                if episode % 100 == 0:
                    print(f"episode: {episode}, total reward: {np.round(np.sum(rewards), decimals = 3)}, average_reward: {np.round(np.mean(all_rewards[-10:]), decimals = 3)}, length: {steps}")
                break
            
            state = new_state
    
    return numsteps, avg_numsteps, all_rewards


def main():
    """
    Function that calls the experiment function, if wanted with different parameters.
    This is used to establish a useful set of parameters.
    We loop over different values of Gamma and the maximum steps.

    gamma_values = [0.6, 0.7, 0.8, 0.9, 1.0]
    max_steps_values = [100, 500, 1000, 5000, 10000] >> 100 is really too small.
    max_episode_num_values = [1000, 2000]

    """
    gamma_values = [0.6, 0.7, 0.8, 0.9, 1.0]
    #gamma_values = [0.9]
    #max_steps_values = [500, 1000, 5000, 10000]
    max_steps_values = [5000, 10000]
    max_episode_num_values = [1000]#[1000, 2000]
    learning_rates = [3e-4]
    #learning_rates = [3e-4, 3e-3, 3e-2]

   # ax.plot(numsteps, label='Number of steps')
    for steps in max_steps_values:
        for episodes in max_episode_num_values:
            fig, ax = plt.subplots()  # Create a new figure and axes object
            ax.set_title(f"REINFORCE Average Steps Plot with {steps} maximum steps")
            ax.set_ylabel('Steps')
            ax.set_xlabel('Episode')
            # Try all gamma values
            for learning_rate in learning_rates:
                for gamma in gamma_values:
                    numsteps, avg_numsteps, all_rewards =  experiment(episodes, steps, gamma, learning_rate)
                    smoothed = savgol_filter(avg_numsteps,30,1)
                    ax.plot(smoothed, label=f'Average steps with gamma = {gamma}') 
            ax.legend()
            plt.savefig(f"REINFORCE_{episodes}_{steps}.png")
            plt.close(fig)  # Close the figure to free up memory
    # max_episode_num = 300
    # max_steps = 10000
    # gamma = 0.9
    # experiment(max_episode_num, max_steps, gamma)

if __name__ == '__main__':
    main()

# EOF