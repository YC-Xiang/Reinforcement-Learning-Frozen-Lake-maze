"""
Brain of the RL agent
including the three algorithms (Monte Carlo, Q-learning, Sarsa)

Hyper-parameters setting:
learning rate: 0.1
discount factor: 0.9
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----Definition of RL model-----
class RL:
    def __init__(self, actions, states):
        self.states = states  # state space
        self.actions = actions  # action space
        self.LearningRate = 0.1  # learning rate
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.1  # e-greedy parameter

        # create the Q-table (index:16 states   columns: 4 actions)
        self.q_table = pd.DataFrame(np.zeros((16, 4)), index=self.states, columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        # Choosing the action according soft e-greedy algorithm

        # 1-e+e/|A(St)|  90% + 2.5% for choosing best q_value
        if np.random.uniform(0, 1) > self.epsilon + self.epsilon / 4:
            state_action = self.q_table.loc[observation, :]  # list the 4 actions of state s

            # if Q_values are equal then choose a random action
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()  # choose the max q_value
        else:
            action = np.random.choice(self.actions)  # left 7.5% for choosing randomly
        return action

    def print_q_table(self):
        # set the parameters of the table
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_rows', 1000)
        pd.set_option('display.width', 1000)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)
        print()
        print('Q-table:')
        print(self.q_table)

    def plot_results(self, steps, q_sum, t_sum):
        # Plot the steps over episodes
        plt.figure()
        plt.plot(np.arange(len(steps)), steps, 'blue', linewidth=1)
        plt.title('Steps over episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Steps')

        # Plot the Q_sum over episodes
        plt.figure()
        plt.plot(np.arange(len(q_sum)), q_sum, 'red', linewidth=1)
        plt.title('Sum of Q_value over episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Sum of Q_value')

        # Plot the Episodes over time
        plt.figure()
        plt.plot(t_sum, np.arange(len(steps)), 'green')
        plt.title('Episodes over Time')
        plt.xlabel('Time(s)')
        plt.ylabel('Episodes')

        plt.show()


# -----Q-learning Algorithm-----
class QLearningTable(RL):
    def __init__(self, actions, states):
        super(QLearningTable, self).__init__(actions, states)

    # Function for updating Q-table
    def update_table(self, s, a, r, s_):
        # Current state in the current position
        q = self.q_table.loc[s, a]

        # Checking if the next state is free or it is obstacle or goal
        if s_ != [150, 150] or s_ != [150, 350] or s_ != [250, 350] or s_ != [350, 50]:
            q_ = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_ = r

        # Updating Q-table with new knowledge
        self.q_table.loc[s, a] += self.LearningRate * (q_ - q)

        return self.q_table.loc[s, a]


# -----Sarsa Algorithm-----
class SarsaTable(RL):
    def __init__(self, actions, states):
        super(SarsaTable, self).__init__(actions, states)

    def update_table(self, s, a, r, s_, a_):
        # Current state in the current position
        q = self.q_table.loc[s, a]

        # Checking if the next state is free or it is obstacle or goal
        if s_ != [150, 150] or s_ != [150, 350] or s_ != [250, 350] or s_ != [350, 50]:
            q_ = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_ = r

        # Updating Q-table with new knowledge
        self.q_table.loc[s, a] += self.LearningRate * (q_ - q)

        return self.q_table.loc[s, a]


# -----MonteCarlo Algorithm-----
class MonteCarloTable(RL):
    def __init__(self, actions, states):
        super(MonteCarloTable, self).__init__(actions, states)

    # calculate discounted return G
    def discounted_rewards(self, rewards):
        current_reward = 0
        discounted_rewards = [0] * len(rewards)
        for t in reversed(range(len(rewards))):
            current_reward = self.gamma * current_reward + rewards[t]
            discounted_rewards[t] = current_reward
        return discounted_rewards

    def update_table(self, s, a, reward_discounted):
        # Current state in the current position
        q = self.q_table.loc[s, a]

        # Updating Q-table with new knowledge
        self.q_table.loc[s, a] += self.LearningRate * (reward_discounted - q)

        return self.q_table.loc[s, a]
