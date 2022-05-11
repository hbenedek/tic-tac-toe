import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tic_env import TictactoeEnv, OptimalPlayer
from tqdm import tqdm

def tuple_to_int(action):
    return action[0] * 3 + action[1]

def int_to_tuple(position):
    return (int(position/3), position % 3)

def play_game(agent1, agent2, env, i, train=True):
    grid, end, __  = env.observe()
    if i % 2 == 0:
        agent1.player = 'X'
        agent2.player = 'O'
    else:
        agent1.player = 'O'
        agent2.player = 'X'
    while end == False:
        if env.current_player == agent1.player:
            move = agent1.act(grid) 
            grid, end, winner = env.step(move, print_grid=False)
        else:
            move = agent2.act(grid)
            grid, end, winner = env.step(move, print_grid=False) 
            if train and not end:
                reward = env.reward(agent1.player)
                agent1.updateQ(grid.copy(),reward)
                
    reward = env.reward(agent1.player)
    agent1.updateQ(grid.copy(),reward)
    return winner, agent1, agent2

def simulate(agent1, agent2, N=500, train=True, bar=True):
    env = TictactoeEnv()
    N_win = 0
    N_loose = 0
    history = []
    
    # set exploration to 0 in test environment
    if not train:
        temp_eps = agent1.epsilon
        temp_explore = agent1.explore
        agent1.epsilon = 0
        agent1.explore = False
        
    for i in tqdm(range(N), disable=not bar):
        # initialize env and players
        env.reset()
        agent1.counter += 1
        # simulation
       
        winner, agent1, agent2 = play_game(agent1, agent2, env, i, train)

        # save results
        if winner == agent1.player:
            N_win += 1
            history.append(1)
        elif winner == agent2.player:
            N_loose += 1
            history.append(-1)
        else:
            history.append(0)
    
    if not train:
        agent1.epsilon = temp_eps
        agent1.explore = temp_explore
           
    return history, (N_win - N_loose) / N, agent1

def learning_evolution(agent1, agent2, N=80):
    m_opts = []
    m_rands = []
    agent_opt = OptimalPlayer(epsilon = 0)
    agent_rand = OptimalPlayer(epsilon = 1)
    
    for i in tqdm(range(N)):
        # training phase
        _, __, agent1 = simulate(agent1, agent2, N=250, train=True, bar=False)
        # testing phase
        _, m_opt, agent1 = simulate(agent1, agent_opt, N=500, train=False, bar=False)
        _, m_rand, agent1 = simulate(agent1, agent_rand, N=500, train=False, bar=False)
        #save results
        m_opts.append(m_opt)
        m_rands.append(m_rand)
           
    return m_opts, m_rands

def grid_to_tensor(grid, player='X'):
    tensor = np.zeros((3,3,2))
    grid1 = grid.copy()
    if player == 'X':
        grid1[grid == -1] = 0
        tensor[:,:,0] = grid1

        grid[grid == 1] = 0
        tensor[:,:,1] = np.abs(grid)
    else: 
        grid1[grid == 1] = 0
        tensor[:,:,0] = np.abs(grid1)

        grid[grid == -1] = 0
        tensor[:,:,1] = grid
    return torch.tensor(tensor, dtype=float).flatten()

def tensor_to_grid(tensor, player='X'):
    return (tensor[:,:,0] + tensor[:,:,1] * (- 1)).numpy() if player == 'X' else (tensor[:,:,0] * (- 1) + tensor[:,:,1]).numpy()
