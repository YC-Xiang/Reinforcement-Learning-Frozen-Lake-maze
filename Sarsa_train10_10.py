"""
Sarsa training for 10*10 frozen lake with random obstacles
take about 9min to finish 2000 episodes training
success rate : 26.25%
The shortest route: 18
The longest route: 323
"""
from env10_10 import Environment
from agent_brain10_10 import SarsaTable
import time


# -----function for the training loop-----
def train():
    steps = []                # record steps of each episode
    Q_sum = []                # summed Q_value for all episodes
    success_times = 0         # success times of all episodes
    t = 0                     # record each episode running  time
    t_sum = []                # record the sum time of training loop

    for episode in range(2000):
        start = time.time()   # start time of each episode
        s = env.reset()       # initial observation
        i = 0                 # update number of steps for each episode
        Q = 0                 # update the reward for each episode

        a = RL.choose_action(str(s))    # choose action based on observation first

        while True:
            env.render()                                      # refresh the environment
            s_, r, done = env.step(a)                         # take the action and get the next observation and reward
            a_ = RL.choose_action(str(s_))                    # take the next action based on next observation
            Q += RL.update_table(str(s), a, r, str(s_), a_)   # learn from the transition and calculate the Q_sum
            s = s_                                            # swap the observations
            a = a_                                            # swap the actions
            i += 1                                            # calculate the number of steps in the current episode

            # if the steps of each episode > 300: break
            if i > 500:
                break

            # when agent reached the goal
            if env.next_state == [475, 475]:                  # when the agent reach the goal coordinate
                success_times += 1                            # record the times of reaching the goal

            # when agent reached the goal or obstacle
            if done:
                steps += [i]                                  # steps of each episode
                Q_sum += [Q]                                  # Q_sum of each episode
                end = time.time()                             # end time of each episode
                t_sum.append(t)                               # append each time of episode into a list
                t += end - start                              # time of each episode
                break

    print('running time:', t)                                 # print the simulation time of the algorithm
    print('success times:', success_times)                    # print the times of reaching the goal
    success_rate = success_times / 2000
    print('Success rate: {:.2%}'.format(success_rate))        # show the success rate

    env.final()                            # show the final route

    RL.print_q_table()                     # show the q_table

    RL.plot_results(steps, Q_sum, t_sum)   # plot the Q_sum and steps over episodes


if __name__ == "__main__":
    # call for the environment
    env = Environment()
    # input the actions, states, hyper-parameters to call for the main algorithm
    RL = SarsaTable(actions=[0, 1, 2, 3],
                    states=['[25.0, 25.0]', '[25.0, 75.0]', '[25.0, 125.0]', '[25.0, 175.0]', '[25.0, 225.0]',
                            '[25.0, 275.0]', '[25.0, 325.0]', '[25.0, 375.0]', '[25.0, 425.0]', '[25.0, 475.0]',
                            '[75.0, 25.0]', '[75.0, 75.0]', '[75.0, 125.0]', '[75.0, 175.0]', '[75.0, 225.0]',
                            '[75.0, 275.0]', '[75.0, 325.0]', '[75.0, 375.0]', '[75.0, 425.0]', '[75.0, 475.0]',
                            '[125.0, 25.0]', '[125.0, 75.0]', '[125.0, 125.0]', '[125.0, 175.0]', '[125.0, 225.0]',
                            '[125.0, 275.0]', '[125.0, 325.0]', '[125.0, 375.0]', '[125.0, 425.0]', '[125.0, 475.0]',
                            '[175.0, 25.0]', '[175.0, 75.0]', '[175.0, 125.0]', '[175.0, 175.0]', '[175.0, 225.0]',
                            '[175.0, 275.0]', '[175.0, 325.0]', '[175.0, 375.0]', '[175.0, 425.0]', '[175.0, 475.0]',
                            '[225.0, 25.0]', '[225.0, 75.0]', '[225.0, 125.0]', '[225.0, 175.0]', '[225.0, 225.0]',
                            '[225.0, 275.0]', '[225.0, 325.0]', '[225.0, 375.0]', '[225.0, 425.0]', '[225.0, 475.0]',
                            '[275.0, 25.0]', '[275.0, 75.0]', '[275.0, 125.0]', '[275.0, 175.0]', '[275.0, 225.0]',
                            '[275.0, 275.0]', '[275.0, 325.0]', '[275.0, 375.0]', '[275.0, 425.0]', '[275.0, 475.0]',
                            '[325.0, 25.0]', '[325.0, 75.0]', '[325.0, 125.0]', '[325.0, 175.0]', '[325.0, 225.0]',
                            '[325.0, 275.0]', '[325.0, 325.0]', '[325.0, 375.0]', '[325.0, 425.0]', '[325.0, 475.0]',
                            '[375.0, 25.0]', '[375.0, 75.0]', '[375.0, 125.0]', '[375.0, 175.0]', '[375.0, 225.0]',
                            '[375.0, 275.0]', '[375.0, 325.0]', '[375.0, 375.0]', '[375.0, 425.0]', '[375.0, 475.0]',
                            '[425.0, 25.0]', '[425.0, 75.0]', '[425.0, 125.0]', '[425.0, 175.0]', '[425.0, 225.0]',
                            '[425.0, 275.0]', '[425.0, 325.0]', '[425.0, 375.0]', '[425.0, 425.0]', '[425.0, 475.0]',
                            '[475.0, 25.0]', '[475.0, 75.0]', '[475.0, 125.0]', '[475.0, 175.0]', '[475.0, 225.0]',
                            '[475.0, 275.0]', '[475.0, 325.0]', '[475.0, 375.0]', '[475.0, 425.0]', '[475.0, 475.0]'],)
    env.after(2000, train)
    env.mainloop()
