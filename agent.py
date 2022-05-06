from abc import abstractclassmethod
import numpy as np
from tic_env import TictactoeEnv, OptimalPlayer
from tqdm import tqdm
from collections import defaultdict
from utils import *
import random

class Agent:
    def __init__(self, player):
        self.player = player
        self.other_player = str(({'X', 'O'} - set(self.player)).pop())

    def change_player(self):
        self.player = str(({'X', 'O'} - set(self.player)).pop())

    def empty(self, grid):
        '''return all empty positions'''
        avail = []
        for i in range(9):
            pos = (int(i/3), i % 3)
            if grid[pos] == 0:
                avail.append(pos)
        return avail
 
    def randomMove(self, grid):
        """ Chose a random move from the available options. """
        avail = self.empty(grid)
        return avail[random.randint(0, len(avail)-1)]

    @abstractclassmethod
    def act(self):
        pass

    @abstractclassmethod
    def updateQ(self):
        pass

class QPlayer(Agent):
    def __init__(self, epsilon=0.2, alpha=0.05, gamma=0.99, player='X', explore=False, e_min=0.1, e_max=0.8, n_star=20000):
        super().__init__(player)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Qvalues = defaultdict(lambda: defaultdict(int))
        self.last_action = 0
        self.last_state = 0
        self.player2value = {'X': 1, 'O': -1}
        self.counter = 0
        self.explore = explore
        self.e_min = e_min
        self.e_max = e_max
        self.n_star = n_star

    def act(self, grid):
        if self.explore:
            self.epsilon = self.decreasing_exploration(self.e_min, self.e_max, self.n_star, self.counter)
        # epsilon greedy policy w.r.t. Q-values
        if random.random() < self.epsilon:
            action = self.randomMove(grid)
            self.last_action = tuple_to_int(action)
        else:
            idx = np.argmax([self.Qvalues[str(grid)][tuple_to_int(pos)] for pos in self.empty(grid)])
            action = self.empty(grid)[idx]
            self.last_action = tuple_to_int(action)

        self.last_state = str(grid)  
        return action

    def updateQ(self, grid, reward):
        future_estimate = max([self.Qvalues[str(grid)][tuple_to_int(pos)] for pos in self.empty(grid)], default=0)
        self.Qvalues[self.last_state][self.last_action] += self.alpha * (reward + self.gamma * future_estimate - self.Qvalues[self.last_state][self.last_action]) 
            
    def decreasing_exploration(self, e_min, e_max, n_star, n):
        return max(e_min, e_max * (1 - n/n_star)) 