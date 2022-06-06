from abc import abstractclassmethod
import numpy as np
from tic_env import TictactoeEnv, OptimalPlayer
from tqdm import tqdm
from collections import defaultdict
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy

class Agent:
    """
    Abstract class for learning agents interacting with TicTacToeEnv, for all kind of agents,
    an act() function should be implemented.
    
     Attributes
     ----------
     explore (bool) : boolean flag to switch on decreasing exploration
     e_min (float) : minimum epsilon value of decreasing exploration
     e_max (float) : maximum epsilon value of decreasing exploration
     n_star (int) : controls the decreasing rate of exploration
    """
    def __init__(self, player, explore, e_min, e_max, n_star):
        self.player = player
        self.other_player = str(({'X', 'O'} - set(self.player)).pop())
        self.explore = explore
        self.e_min = e_min
        self.e_max = e_max
        self.n_star = n_star

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

    def copy(self):
        """ Creates a copy of the agent. """
        return copy.deepcopy(self)

    def decreasing_exploration(self, e_min, e_max, n_star, n):
        """ Returns the current value of epsilon after n rounds of simulated games. """
        return max(e_min, e_max * (1 - n/n_star)) 

    @abstractclassmethod
    def act(self):
        """ Abstarct class method for the agent's policy execution. """
        pass

class QPlayer(Agent):
     """
     Tabular Q-learning agent class
    
     Attributes
     ----------
     epsilon (float) : fixed exploration rate
     alpha (float) : learning rate
     gamma (float) : discount factor
     player (string) : 'X' or 'O'
     explore (bool) : boolean flag to switch on decreasing exploration
     e_min (float) : minimum epsilon value of decreasing exploration
     e_max (float) : maximum epsilon value of decreasing exploration
     n_star (int) : controls the decreasing rate of exploration
    """
    def __init__(self, epsilon=0.2, alpha=0.05, gamma=0.99, player='X', explore=False, e_min=0.1, e_max=0.8, n_star=20000):
        super().__init__(player, explore, e_min, e_max, n_star)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Qvalues = defaultdict(lambda: defaultdict(int))
        self.last_action = 0
        self.last_state = 0
        self.player2value = {'X': 1, 'O': -1}
        self.counter = 0

    def act(self, grid):
        """ epsilon greedy ploicy with respect to the Q-values. """
        if self.explore:
            self.epsilon = self.decreasing_exploration(self.e_min, self.e_max, self.n_star, self.counter)
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
        """ For a given (state, action, reward, next state) tuple, updates the Q-value of (state, action)"""
        future_estimate = max([self.Qvalues[str(grid)][tuple_to_int(pos)] for pos in self.empty(grid)], default=0)
        self.Qvalues[self.last_state][self.last_action] += self.alpha * (reward + self.gamma * future_estimate - self.Qvalues[self.last_state][self.last_action]) 

    def updateQ_self(self, grid, reward, s, a):
        """ For a given (state, action, reward, next state) tuple, updates the Q-value of (state, action) (second implementation)"""
        future_estimate = max([self.Qvalues[str(grid)][tuple_to_int(pos)] for pos in self.empty(grid)], default=0)
        self.Qvalues[s][a] += self.alpha * (reward + self.gamma * future_estimate - self.Qvalues[s][a])

class DeepAgent(Agent):
     """
     Agnet learning with Deep-Q network (DQN) and memory buffer
    
     Attributes
     ----------
     epsilon (float) : fixed exploration rate
     gamma (float) : discount factor
     buffer (int) : size of the memory buffer
     explore (bool) : boolean flag to switch on decreasing exploration
     player (string) : 'X' or 'O'
     e_min (float) : minimum epsilon value of decreasing exploration
     e_max (float) : maximum epsilon value of decreasing exploration
     n_star (int) : controls the decreasing rate of exploration
    """
        
    def __init__(self, epsilon=0.2, gamma=0.99, buffer=10000, batch=64, update_target=500, explore=False, player='X', e_min=0.1, e_max=0.8, n_star=20000):
        super().__init__(player, explore, e_min, e_max, n_star)
        self.gamma = gamma
        self.buffer = buffer
        self.batch = batch
        self.epsilon = epsilon
        self.update_target = update_target
        self.counter = 0
        self.DQN = DeepQNetwork()
        self.target_network = DeepQNetwork()
        self.memory = Memory(self.buffer)
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.losses = []
        self.avg_losses = []
        self.m_opts = []
        self.m_rands = [] 

    def act(self, grid):
        """ Epsilon greedy w.r.t. Qvalues generated by Deep Network. """
        if self.explore:
            self.epsilon = self.decreasing_exploration(self.e_min, self.e_max, self.n_star, self.counter)
        if random.random() < self.epsilon:
            action = self.randomMove(grid)
            self.last_action = tuple_to_int(action)
        else:
            with torch.no_grad():
                action = self.DQN(grid_to_tensor(grid.copy(), self.player)).argmax().item()
                self.last_action = action

        self.last_state = grid_to_tensor(grid.copy(), self.player)  
        return action

    def play_game(self, agent, env, i, train=True):
        """ 
        Given an opponent agent, simulates a playout of Tic Tac Toe, 
        if train boolean flag is switched on, encountered (state, action, reward, next state) tuples are saved to memory buffer.
        """
        #initialization 
        grid, end, __  = env.observe()
        if i % 2 == 0:
            self.player = 'X'
            agent.player = 'O'
        else:
            self.player = 'O'
            agent.player = 'X'
            
        # simulation
        while end == False:
            if env.current_player == self.player:
                move = self.act(grid) 
                grid, end, winner = env.step(move, print_grid=False)
            else:
                move = agent.act(grid)
                grid, end, winner = env.step(move, print_grid=False) 
                if train and not end:
                    reward = env.reward(self.player)
                    self.memory.update(self.last_state, self.last_action, reward, grid_to_tensor(grid.copy(), self.player))
                    self.optimize_model()
        if train: 
            reward = env.reward(self.player)
            self.memory.update(self.last_state, self.last_action, reward, grid_to_tensor(grid.copy(), self.player))
            self.optimize_model()
        return winner

    def play_n_games(self, agent, env, episodes, train=True):
        """ Wrapper function around play_game to simulate n number of games. """
        N_win = 0
        N_loose = 0 
        for j in range(episodes):
            env.reset()
            winner = self.play_game(agent, env, 0, train=False)

            if winner == self.player:
                N_win += 1
            elif winner == agent.player:
                N_loose += 1

        return (N_win - N_loose) / episodes 

    
    def optimize_model(self):
        """ Update DQN parameters based on sampled Q-values from the buffer"""
        if self.memory.memory_used >= self.batch:
            # train phase

            # sample mini-batch from memory
            state, action, reward, next_state = self.memory.sample(self.batch)
            
            self.DQN.optimizer.zero_grad()
            # calculate "prediction" Q values
            output = self.DQN(state)
            y_pred = output.gather(1, action.long().view((self.batch, 1))).view(-1)
            
            # calculate "target" Q-values from Q-Learning update rule
            mask = (~reward.abs().bool()).int() #invert rewards {0 -> 1, {-1,1} -> 0}
            y_target =  reward + self.gamma * mask * self.target_network(next_state).max(dim=1)[0].detach()

            # forward + backward + optimize
            loss = self.DQN.criterion(y_pred, y_target)
            loss.backward()
            for param in self.DQN.parameters():
                param.grad.data.clamp_(-1, 1)
            self.DQN.optimizer.step()
            self.losses.append(loss.detach().numpy())


    def learn(self, agent, N=20000, test_phase=250, self_practice=False, save_avg_losses=False):
        """ 
        Main entry point of learning.
        
        Attributes
        ----------
        agent (OptimalPlayer) : opponent agent for which we train against
        N (int) : number of simulated games
        test_phase (int) : after every number of games defined by this variable 500 test games are played 
                            against Opt(1) and Opt(0) and results are saved
        self_practice (bool) : boolean flag to switch on learning with self-practice
        save_avg_losses (bool) : boolean flag to switch on saving losses against training agent
        """
        history = []
        env = TictactoeEnv()
        opt_agent = OptimalPlayer(0)
        rand_agent = OptimalPlayer(1)
        agent_copy = self.copy()
        for i in tqdm(range(0, N)):
            env.reset()
            self.counter += 1
            if self_practice:
                winner = self.self_practice(env, agent_copy, train=True) 
                
                # update target network
                if i % self.update_target == 0:
                    self.target_network.load_state_dict(self.DQN.state_dict())
                    agent_copy.target_network.load_state_dict(self.DQN.state_dict())         
          
            else:
                winner = self.play_game(agent, env, i, train=True) 

                # update target network
                if i % self.update_target == 0:
                    self.target_network.load_state_dict(self.DQN.state_dict())
            
            # save results
            if winner == self.player:
                history.append(1)
            elif winner == agent.player:
                history.append(-1)
            else:
                history.append(0)

            if i % test_phase == 0:
                self.simulate_test_phase(opt_agent, rand_agent, env)

            if save_avg_losses and i % 250 == 0:
                self.avg_losses.append(np.mean(np.array(self.losses)))
                self.losses = []


        return history

    def simulate_test_phase(self, opt_agent, rand_agent, env, episodes=500):  
        """ Simulate episodes number of games (500) against Opt(1) and Opt(0) agents without training"""
        # set exploration to 0 in test environment
        temp_eps = self.epsilon
        temp_explore = self.explore
        self.epsilon = 0
        self.explore = False   

        # play test games against optimal and random opponent
        m_opt = self.play_n_games(opt_agent, env, episodes, train=False)
        m_rand = self.play_n_games(rand_agent, env, episodes, train=False)
        self.m_opts.append(m_opt)
        self.m_rands.append(m_rand)

        # set exploration back to  
        self.epsilon = temp_eps
        self.explore = temp_explore


    def self_practice(self, env, agent_copy, train=True):
        """ Game simulation for self-practice using a copy of the agent. """
        grid, end, __  = env.observe()
        first_move = True
        self.player = 'X'
        agent_copy.player = 'O'
        while end == False:
            agent_copy.DQN.load_state_dict(self.DQN.state_dict())
            self.optimize_model() 
            if env.current_player == 'X':
                move = self.act(grid)
                grid, end, winner = env.step(move, print_grid=False)   
                if train and not first_move and winner != 'O':
                    reward = env.reward('O') 
                    self.memory.update(agent_copy.last_state, agent_copy.last_action, reward, grid_to_tensor(grid.copy(), 'O'))
                first_move = False
            else:
                move = agent_copy.act(grid)            
                grid, end, winner = env.step(move, print_grid=False)            
                if train and winner != 'X':
                    reward = env.reward('X') 
                    self.memory.update(self.last_state, self.last_action, reward, grid_to_tensor(grid.copy(), 'X'))     
                    
        
        reward = env.reward('X')
        self.memory.update(self.last_state, self.last_action, reward, grid_to_tensor(grid.copy(), 'X'))
        reward = env.reward('O') 
        self.memory.update(agent_copy.last_state, agent_copy.last_action, reward, grid_to_tensor(grid.copy(), 'O'))
        agent_copy.DQN.load_state_dict(self.DQN.state_dict())
        self.optimize_model()
        return winner

class DeepQNetwork(nn.Module):
     """ 
     Class for implementing the neural network rchitecture for learning Q-values
        
     Attributes
     ----------
     alpha (float) : learning rate of the optimizer
     delta (float) : cutoff value of the Huber loss function
     hidden_neurons : number of hidden neurons on the two hidden layers
     """
    def __init__(self, alpha=5*1e-4, delta=1, hidden_neurons=128):
        super(DeepQNetwork, self).__init__()
        self.linear1 = nn.Linear(18, hidden_neurons)
        self.linear2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.linear3 = nn.Linear(hidden_neurons, 9)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.criterion = nn.SmoothL1Loss()

    def forward(self, x):
        x = x.float()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Memory:
    """
    Class for implementing the memory buffer with a given size
    """
    def __init__(self, size):
        self.size = size
        self.reset_memory()

    def update(self, last_state, last_action, reward, next_state):
        """ Updates the buffer with tuple (state, action, reward, next state) in a rolling window fashion. """
        if isinstance(last_action, int) and isinstance(last_state, torch.Tensor):
            if self.memory_used < self.size:
                    self.memory_used += 1
            self.memory_s[self.position] = last_state
            self.memory_a[self.position] = last_action
            self.memory_r[self.position] = reward
            self.memory_ns[self.position] = next_state
            self.position = (self.position + 1) % self.size

    def sample(self, batch):
        """ Samples a batch from memory for learning. """
        idx = torch.randperm(self.memory_used)[:batch].long()
        return self.memory_s[idx], self.memory_a[idx], self.memory_r[idx], self.memory_ns[idx]

    def reset_memory(self):
        """ Empties and initializes memory. """
        self.memory_s = torch.empty((self.size, 18))
        self.memory_a = torch.empty((self.size))
        self.memory_r = torch.empty((self.size))
        self.memory_ns = torch.empty((self.size, 18))
        self.position = 0
        self.memory_used = 0

   
