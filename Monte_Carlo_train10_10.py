"""
Monte Carlo training for 10*10 frozen lake with random obstacles
take about 3min to finish 2000 episodes training
success rate : 38.45%
The shortest route: 18
The longest route: 420
"""
from env10_10 import Environment
from agent_brain10_10 import MonteCarloTable
import numpy as np
import time


# -----function for generate one episode-----
def monte_carlo_episode():
    episode = []                          # store (s,a,r) of each episode
    s = env.reset()                       # initial observation
    j = 0                                 # if the agent reach the goal j+=1
    t = 0                                 # time of each episode
    start = time.time()                   # record the start time

    while True:
        env.render()                      # refresh the environment
        a = RL.choose_action(str(s))      # choose the action based on observation
        s_, r, done = env.step(a)         # take an action and get the next observation and reward
        episode.append([s, a, r])         # append (St,At+1,Rt+1) into an numpy array
        s = s_                            # swap the observations

        # when agent reach the goal coordinate
        if env.next_state == [475, 475]:
            j += 1                        # j record the times of reaching the goal

        # when agent reached the obstacle or goal
        if done:
            end = time.time()             # record the end time
            t += end - start              # time of each episode
            env.reset()                   # reset the agent
            break
    return episode, t, j


def first_visit(episode):
    episode11 = [i[0] for i in episode]         # extract states of episode
    episode22 = [i[1] for i in episode]         # extract actions of episode
    episode33 = [i[2] for i in episode]         # extract rewards of episode

    # delete the same states and their corresponding actions and rewards
    for i in range(len(episode11)-2, 0, -1):
        for j in range(0, i):
            if episode11[i] == episode11[j]:
                del episode11[i]
                del episode22[i]
                del episode33[i-1]
                break
    return episode11, episode22, episode33


# -----function for the training loop-----
def train():
    reward_history = []         # record average return (St,At)
    steps = []                  # record steps of each episode
    t_ = 0                      # record sum_time
    t_sum = []                  # sum time of training loop
    success_times = 0           # record the times reaching the goal
    Q_sum = []                  # record the Q_sum

    # loop for all episodes
    for i in range(2000):
        episode, t, j = monte_carlo_episode()
        episode1 = [i[0] for i in episode]   # record the numbers of all steps
        steps.append(len(episode1))          # append numbers of states into a list

        t_ += t                              # summed time
        t_sum.append(t_)                     # append summed time into a list for plot
        success_times += j                   # success times of reaching the goal

        episode11, episode22, episode33 = first_visit(episode)  # first visit MC

        rewards = episode33                  # extract the column: reward
        r = RL.discounted_rewards(rewards)   # calculate the discounted reward
        Q_sum.append(sum(r))                 # calculate the summed Q_value G<-gamma*G+R(t+1)

        # loop (update the Q table) for one episode
        for k in range(len(episode11)):
            RL.update_table(str(episode11[k]), episode22[k], r[k])
        reward_history.append(np.mean(r))               # append average return (St,At) into a list

    print('success times:', success_times)              # print the success times
    success_rate = success_times / 2000
    print('Success rate: {:.2%}'.format(success_rate))  # show the success rate
    print('running time:', t_)                          # print the simulation time of the algorithm

    env.final()                                         # show the final route

    RL.print_q_table()                                  # show the Q-table

    RL.plot_results(steps, Q_sum, t_sum)                # plot the Q_sum and steps over episodes


if __name__ == "__main__":
    # call for the environment
    env = Environment()
    # input the actions and states to call for the main algorithm
    RL = MonteCarloTable(actions=[0, 1, 2, 3],
                         states=['[25.0, 25.0]',  '[25.0, 75.0]',  '[25.0, 125.0]', '[25.0, 175.0]', '[25.0, 225.0]',
                                 '[25.0, 275.0]', '[25.0, 325.0]', '[25.0, 375.0]', '[25.0, 425.0]', '[25.0, 475.0]',
                                 '[75.0, 25.0]',  '[75.0, 75.0]',  '[75.0, 125.0]', '[75.0, 175.0]', '[75.0, 225.0]',
                                 '[75.0, 275.0]', '[75.0, 325.0]', '[75.0, 375.0]', '[75.0, 425.0]', '[75.0, 475.0]',
                                 '[125.0, 25.0]', '[125.0, 75.0]', '[125.0, 125.0]', '[125.0, 175.0]', '[125.0, 225.0]',
                                 '[125.0, 275.0]','[125.0, 325.0]','[125.0, 375.0]', '[125.0, 425.0]', '[125.0, 475.0]',
                                 '[175.0, 25.0]', '[175.0, 75.0]', '[175.0, 125.0]', '[175.0, 175.0]', '[175.0, 225.0]',
                                 '[175.0, 275.0]','[175.0, 325.0]','[175.0, 375.0]', '[175.0, 425.0]', '[175.0, 475.0]',
                                 '[225.0, 25.0]', '[225.0, 75.0]', '[225.0, 125.0]', '[225.0, 175.0]', '[225.0, 225.0]',
                                 '[225.0, 275.0]','[225.0, 325.0]','[225.0, 375.0]', '[225.0, 425.0]', '[225.0, 475.0]',
                                 '[275.0, 25.0]', '[275.0, 75.0]', '[275.0, 125.0]', '[275.0, 175.0]', '[275.0, 225.0]',
                                 '[275.0, 275.0]','[275.0, 325.0]','[275.0, 375.0]', '[275.0, 425.0]', '[275.0, 475.0]',
                                 '[325.0, 25.0]', '[325.0, 75.0]', '[325.0, 125.0]', '[325.0, 175.0]', '[325.0, 225.0]',
                                 '[325.0, 275.0]','[325.0, 325.0]','[325.0, 375.0]', '[325.0, 425.0]', '[325.0, 475.0]',
                                 '[375.0, 25.0]', '[375.0, 75.0]', '[375.0, 125.0]', '[375.0, 175.0]', '[375.0, 225.0]',
                                 '[375.0, 275.0]','[375.0, 325.0]','[375.0, 375.0]', '[375.0, 425.0]', '[375.0, 475.0]',
                                 '[425.0, 25.0]', '[425.0, 75.0]', '[425.0, 125.0]', '[425.0, 175.0]', '[425.0, 225.0]',
                                 '[425.0, 275.0]','[425.0, 325.0]','[425.0, 375.0]', '[425.0, 425.0]', '[425.0, 475.0]',
                                 '[475.0, 25.0]', '[475.0, 75.0]', '[475.0, 125.0]', '[475.0, 175.0]', '[475.0, 225.0]',
                                 '[475.0, 275.0]','[475.0, 325.0]','[475.0, 375.0]', '[475.0, 425.0]', '[475.0, 475.0]'],)
    env.after(2000, train)
    env.mainloop()
