"""
10*10 frozen lake environment with 25 random obstacles

Action space:{'up':0  'down':1  'left':2  'right':3}
State space:{'coordinate of tk.canvas':100 states}
Reward:{'goal':+1  'obstacle':-1  others:0}

Environment is based on:
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
using tkinter to build the environment
"""

import numpy as np
import tkinter as tk
import time
from PIL import Image, ImageTk  # For adding images
import random

# Environment size
pixels = 50  # pixels
env_height = 10  # grid height
env_width = 10  # grid width


# A class to represent the environment
class Environment(tk.Tk, object):
    def __init__(self):
        super(Environment, self).__init__()
        self.obstacle_all = []
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = 4
        self.title('RL Q-learning. Xiang Yucheng')
        self.build_environment()

        # Dictionaries to draw the final route
        self.d = {}
        self.f = {}

        # Key for the dictionaries
        self.i = 0

        # Writing the final dictionary first time
        self.c = True

        # Showing the steps for longest found route
        self.longest = 0

        # Showing the steps for the shortest route
        self.shortest = 0

    # Function to build the environment
    def build_environment(self):
        self.canvas = tk.Canvas(self, bg='white', height=env_height * pixels, width=env_width * pixels)

        # Creating grid lines
        for column in range(0, env_width * pixels, pixels):
            x0, y0, x1, y1 = column, 0, column, env_height * pixels
            self.canvas.create_line(x0, y0, x1, y1, fill='grey')
        for row in range(0, env_height * pixels, pixels):
            x0, y0, x1, y1 = 0, row, env_height * pixels, row
            self.canvas.create_line(x0, y0, x1, y1, fill='grey')

        # Creating Obstacles
        img_obstacle1 = Image.open("images/obstacle10_10.png")
        self.obstacle1_object = ImageTk.PhotoImage(img_obstacle1)

        # set 25 obstacles randomly
        random.seed(1)                  # generate fake random numbers
        store = []                      # store the coordinates of canvas
        for m in range(10):
            for n in range(10):
                store.append([m, n])
        del store[0]                       # prevent the agent stuck
        del store[0]                       # prevent the agent stuck
        del store[97]                      # prevent the agent stuck
        del store[96]                      # prevent the agent stuck
        store1 = random.sample(store, 25)  # sample 25 coordinates randomly
        i = 0
        while i < 25:
            self.obstacle = self.canvas.create_image(pixels * (store1[i][0] + 0.5),
                                                     pixels * (store1[i][1] + 0.5),
                                                     anchor='center', image=self.obstacle1_object)
            self.obstacle_all.append(self.obstacle)
            i += 1

        # Create the Goal
        img_goal = Image.open("images/goal10_10.png")
        self.goal_object = ImageTk.PhotoImage(img_goal)
        self.goal = self.canvas.create_image(pixels * 9.5, pixels * 9.5, anchor='center', image=self.goal_object)

        # Create the Robot
        img_robot = Image.open("images/robot10_10.png")
        self.robot = ImageTk.PhotoImage(img_robot)
        self.agent = self.canvas.create_image(pixels * 0.5, pixels * 0.5, anchor='center', image=self.robot)

        self.canvas.pack()

    # Function to reset the environment and start new Episode
    def reset(self):
        self.update()
        time.sleep(0.001)

        # Updating agent
        self.canvas.delete(self.agent)
        self.agent = self.canvas.create_image(pixels * 0.5, pixels * 0.5, anchor='center', image=self.robot)

        # # Clearing the dictionary and the i
        self.d = {}
        self.i = 0

        # Return observation
        return self.canvas.coords(self.agent)

    # Function to get the next observation and reward by doing next step
    def step(self, action):
        # Current state of the agent
        state = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])

        # Updating next state according to the action
        # Action 'up'
        if action == 0:
            if state[1] >= pixels:
                base_action[1] -= pixels
        # Action 'down'
        elif action == 1:
            if state[1] < (env_height - 1) * pixels:
                base_action[1] += pixels
        # Action right
        elif action == 2:
            if state[0] < (env_width - 1) * pixels:
                base_action[0] += pixels
        # Action left
        elif action == 3:
            if state[0] >= pixels:
                base_action[0] -= pixels

        # Moving the agent according to the action
        self.canvas.move(self.agent, base_action[0], base_action[1])

        # Writing in the dictionary coordinates of found route
        self.d[self.i] = self.canvas.coords(self.agent)

        # Updating next state
        self.next_state = self.d[self.i]
        # Updating key for the dictionary
        self.i += 1

        # Calculating the reward for the agent
        if self.next_state == self.canvas.coords(self.goal):
            reward = 1
            done = True

            # Filling the dictionary first time
            if self.c:
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]
                self.c = False
                self.longest = len(self.d)
                self.shortest = len(self.d)

            # Checking if the currently found route is shorter
            if len(self.d) < len(self.f):
                # Saving the number of steps for the shortest route
                self.shortest = len(self.d)
                # Clearing the dictionary for the final route
                self.f = {}
                # Reassigning the dictionary
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]

            # Saving the number of steps for the longest route
            if len(self.d) > self.longest:
                self.longest = len(self.d)

        elif self.next_state in [self.canvas.coords(self.obstacle_all[0]),
                                 self.canvas.coords(self.obstacle_all[1]),
                                 self.canvas.coords(self.obstacle_all[2]),
                                 self.canvas.coords(self.obstacle_all[3]),
                                 self.canvas.coords(self.obstacle_all[4]),
                                 self.canvas.coords(self.obstacle_all[5]),
                                 self.canvas.coords(self.obstacle_all[6]),
                                 self.canvas.coords(self.obstacle_all[7]),
                                 self.canvas.coords(self.obstacle_all[8]),
                                 self.canvas.coords(self.obstacle_all[9]),
                                 self.canvas.coords(self.obstacle_all[10]),
                                 self.canvas.coords(self.obstacle_all[11]),
                                 self.canvas.coords(self.obstacle_all[12]),
                                 self.canvas.coords(self.obstacle_all[13]),
                                 self.canvas.coords(self.obstacle_all[14]),
                                 self.canvas.coords(self.obstacle_all[15]),
                                 self.canvas.coords(self.obstacle_all[16]),
                                 self.canvas.coords(self.obstacle_all[17]),
                                 self.canvas.coords(self.obstacle_all[18]),
                                 self.canvas.coords(self.obstacle_all[19]),
                                 self.canvas.coords(self.obstacle_all[20]),
                                 self.canvas.coords(self.obstacle_all[21]),
                                 self.canvas.coords(self.obstacle_all[22]),
                                 self.canvas.coords(self.obstacle_all[23]),
                                 self.canvas.coords(self.obstacle_all[24])]:
            reward = -1
            done = True

            # Clearing the dictionary and the i
            self.d = {}
            self.i = 0

        else:
            reward = 0
            done = False

        return self.next_state, reward, done

    # Function to refresh the environment
    def render(self):
        time.sleep(0.001)
        self.update()

    # Function for showing the found route
    def final(self):
        # Deleting the agent at the end
        self.canvas.delete(self.agent)

        # Showing the number of steps
        print('The shortest route:', self.shortest)
        print('The longest route:', self.longest)

        # Creating initial point
        self.initial_point = self.canvas.create_oval(15, 15, 35, 35, fill='red', outline='red')

        # Filling the route
        for j in range(len(self.f)):
            # Showing the coordinates of the final route
            print(self.f[j])
            self.track = self.canvas.create_oval(
                self.f[j][0] - 10, self.f[j][1] - 10,
                self.f[j][0] + 10, self.f[j][1] + 10,
                fill='red', outline='red')


# This shows the static environment without running algorithm
if __name__ == '__main__':
    env = Environment()
    env.mainloop()
