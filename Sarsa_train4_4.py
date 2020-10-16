"""
Sarsa training for 4*4 frozen lake
take about 50s to finish 300 episodes training
success rate : 83.30%
The shortest route: 6
The longest route: 82
"""
from env4_4 import Environment
from agent_brain4_4 import SarsaTable
import time


# -----function for the training loop-----
def train():
    steps = []               # record steps of each episode
    Q_sum = []               # summed Q_value for all episode
    success_times = 0        # success times of all episodes
    t = 0                    # record each episode running time
    t_sum = []               # record the sum time of training loop

    for episode in range(300):
        start = time.time()  # Start time of each episode
        s = env.reset()      # initial observation
        i = 0                # update number of steps for each episode
        Q = 0                # update the Q for each episode

        a = RL.choose_action(str(s))        # choose action based on observation first

        while True:
            env.render()                                      # refresh the environment
            s_, r, done = env.step(a)                         # take the action and get the next observation and reward
            a_ = RL.choose_action(str(s_))                    # take the next action based on next observation
            Q += RL.update_table(str(s), a, r, str(s_), a_)   # learn from the transition and calculate the Q_sum
            s = s_                                            # swap the observations
            a = a_                                            # swap the actions
            i += 1                                            # calculate the number of steps in the current episode

            # When agent reached the goal or obstacle
            if env.next_state == [350, 350]:                  # when agent reach the goal coordinate
                success_times += 1                            # when the agent reach the goal +1

            if done:
                steps += [i]                             # steps of each episode
                Q_sum += [Q]                             # Q_sum of each episode
                end = time.time()                        # end time of each episode
                t_sum.append(t)                          # append each time of episode into a list
                t += end - start                         # time of each episode
                break

    print('running time:', t)                            # print the simulation time of the algorithm
    print('success times:', success_times)               # print the success times
    success_rate = success_times / 300
    print('Success rate: {:.2%}'.format(success_rate))   # show the success rate

    env.final()                            # show the final route

    RL.print_q_table()                    # show the Q-table

    RL.plot_results(steps, Q_sum, t_sum)  # plot the Q_sum and steps over episodes


if __name__ == "__main__":
    # call for the environment
    env = Environment()
    # call for the main algorithm
    RL = SarsaTable(actions=[0, 1, 2, 3],
                    states=['[50.0, 50.0]', '[50.0, 150.0]', '[50.0, 250.0]', '[50.0, 350.0]',
                            '[150.0, 50.0]', '[150.0, 150.0]', '[150.0, 250.0]', '[150.0, 350.0]',
                            '[250.0, 50.0]', '[250.0, 150.0]', '[250.0, 250.0]', '[250.0, 350.0]',
                            '[350.0, 50.0]', '[350.0, 150.0]', '[350.0, 250.0]', '[350.0, 350.0]'])
    env.after(300, train)
    env.mainloop()
