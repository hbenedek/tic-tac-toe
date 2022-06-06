import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tic_env import TictactoeEnv, OptimalPlayer
from tqdm import tqdm

def tuple_to_int(action):
    """ Converts the tuple encoding of a board position to integer value. """
    return action[0] * 3 + action[1]

def int_to_tuple(position):
    """ Converts the integer encoding of a board position to tuple representation. """
    return (int(position/3), position % 3)

def play_game(agent1, agent2, env, i, train=True):
    """ Simulates a Tic Tac Toe game between two Agents/OPtimalPlayers. """
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
                
    if train:
        reward = env.reward(agent1.player)
        agent1.updateQ(grid.copy(),reward)
    return winner

def simulate(agent1, agent2, N=500, train=True, bar=True, self_practice=False):
    """ Simulates N number of games against other agent or self_practice depending on boolean flag. """
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

        if self_practice:
            self_play_game(agent1, agent2, env, train)
        else:
            winner = play_game(agent1, agent2, env, i, train)

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
           
    return history, (N_win - N_loose) / N

def learning_evolution(agent1, agent2, N=80, self_practice=False):
    """ 
    Main entry point of learning N defines the number of iterations in learning, one iteration consists of a learning phase 
    (250 games against other agent or learning with self practice)
    and a testing phase (500 game against Opt(0) and 500 games against Opt(1)).
    """
    m_opts = []
    m_rands = []
    agent_opt = OptimalPlayer(epsilon=0)
    agent_rand = OptimalPlayer(epsilon=1)
    
    for i in tqdm(range(N)):
        # training phase
        _, __ = simulate(agent1, agent2, N=250, train=True, bar=False, self_practice=self_practice)
        # testing phase
        _, m_opt = simulate(agent1, agent_opt, N=500, train=False, bar=False)
        _, m_rand = simulate(agent1, agent_rand, N=500, train=False, bar=False)
        #save results
        m_opts.append(m_opt)
        m_rands.append(m_rand)
           
    return m_opts, m_rands

def self_play_game(agent1, agent2, env, train=True):
    """ Implementation of game simulation with self practice. """
    #agent1 and agent2 both are effectively the same agent. We keep this naming to be conistent with previous questions
    grid, end, __  = env.observe()
    first_move = True
    
    while end == False:
        if env.current_player == 'X':
            move = agent1.act(grid)
            agent1_last_action = agent1.last_action
            agent1_last_state = agent1.last_state
            grid, end, winner = env.step(move, print_grid=False)            

            if train and not first_move:
                reward = env.reward('O') #Reward of agent 2
                agent1.updateQ_self(grid.copy(),reward, agent2_last_state, agent2_last_action)
                
            first_move = False
                
        else:
            move = agent2.act(grid)
            agent2_last_action = agent2.last_action
            agent2_last_state = agent2.last_state
            grid, end, winner = env.step(move, print_grid=False)            
            
            if train:
                reward = env.reward('X') #Reward of agent 1
                agent2.updateQ_self(grid.copy(),reward, agent1_last_state, agent1_last_action)
                
    if winner == 'X' or winner is None:
        reward = env.reward('X')
        agent2.updateQ_self(grid.copy(),reward, agent1_last_state, agent1_last_action)
    if winner == 'O':
        reward = env.reward('O') 
        agent1.updateQ_self(grid.copy(),reward, agent2_last_state, agent2_last_action)

def grid_to_tensor(grid, player='X'):
    """ Converts grid to torch tensor. """
    tensor = np.zeros((3,3,2))
    grid1 = grid.copy()
    grid2 = grid.copy()
    if player == 'X':
        grid1[grid == -1] = 0
        tensor[:,:,0] = grid1

        grid2[grid == 1] = 0
        tensor[:,:,1] = np.abs(grid2)
    else: 
        grid1[grid == 1] = 0
        tensor[:,:,0] = np.abs(grid1)

        grid2[grid == -1] = 0
        tensor[:,:,1] = grid2
    return torch.tensor(tensor, dtype=float).flatten()

def tensor_to_grid(tensor, player='X'):
    """ Converts torch tensor to grid. """
    return (tensor[:,:,0] + tensor[:,:,1] * (- 1)).numpy() if player == 'X' else (tensor[:,:,0] * (- 1) + tensor[:,:,1]).numpy()

def swap_positions(tensor):
    """ Given a tensor representation of the board, changes the encoding of 'X' and 'O'."""
    return torch.index_select(tensor, 2, torch.LongTensor([1,0]))

def get_training_time(opts, rands):
    """ 
    Given the results of a learning evolution, it returns the time required to achieve 80% of maximum performance.
    """
    max_opt = (max(opts) + 1) * 0.8 - 1
    max_rand = max(rands) * 0.8
    for i, metric in enumerate(opts):
        if metric > max_opt:
            opt_idx = i
            break

    for i, metric in enumerate(rands):
        if metric > max_rand:
            rand_idx = i
            break

    return opt_idx, rand_idx


