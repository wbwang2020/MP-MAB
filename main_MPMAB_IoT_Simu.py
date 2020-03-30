# -*- coding: utf-8 -*-
"""
@author: Wenbo Wang

[Wang2020] Wenbo Wang, Amir Leshem, Dusit Niyato and Zhu Han, "Decentralized Learning for Channel 
Allocation inIoT Networks over Unlicensed Bandwidth as aContextual Multi-player Multi-armed Bandit Game"

License:
This program is licensed under the GPLv2 license. If you in any way use this code for research 
that results in publications, please cite our original article listed above.
 
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
See the GNU General Public License for more details.
"""

# Used for the simulations in the paper "Decentralized Learning for Channel Allocation in IoT Networks over Unlicensed  
# Bandwidth as a Contextual Multi-player Multi-armed Bandit Game", by Wenbo Wang et al.
# This file is the main entrance of the simulations regarding the network performance vs. network scale.

__author__ = "Wenbo Wang"

# This file is the main entrance of the simulations for algorithm performance w.r.t. network sizes.

__author__ = "Wenbo Wang"

import numpy as np
import pandas as pd

import time
import datetime
import sys
#import argparse

from GameEvaluator import AlgEvaluator
from plotutils import plot_data_frame

from envutils import Struct as Section

def simulation_execution(alg_engine, game_config, player_number, game_horizon, simu_rounds, flag_parallel=False):        
    """
    simulation_execution() is the main body of the MP-MAP algorithm simulations
    """    
    
#    print("number of arms: {}, number of players: {}".format(alg_engine.nbArms, alg_engine.nbPlayers))

    #add algorithms
    for alg_id in range(len(game_config.alg_types)):
        alg_engine.add_algorithm(algo_type=game_config.alg_types[alg_id], 
                                 custome_params=game_config.alg_configs[alg_id])
    
    if flag_parallel == True:
        simulation_results = alg_engine.play_repeated_game_parallel([game_horizon], simulation_rounds=simu_rounds,
                                                                flag_progress_bar=True)
    else:
        # for large network, we use seuqnecial processing in order to avoid overwhelming the memory 
        simulation_results = alg_engine.play_repeated_game([game_horizon], simulation_rounds=simu_rounds,
                                                                flag_progress_bar=True)
            
    
    network_size_indicator_series = []
    alg_indicator_series = []
    reward_series = np.array([])
    switching_series = np.array([])
    collision_series = np.array([])
    
#    print("size of simulation results, rewards: {}".format(np.shape(simulation_results['reward_series'])))
#    print("length of simulation_results: {}".format(len(simulation_results['algorithm_name'])))
    
    for alg_id in range(len(simulation_results['algorithm_name'])):
         avg_rewards = simulation_results['reward_series'][alg_id, :] / game_horizon                  
         switching = simulation_results['switching_count_series'][alg_id, :]
         collisions = simulation_results['collision_series'][alg_id, :]
         
         network_sizes = np.zeros(avg_rewards.shape)
         network_sizes[:] = player_number
                
         reward_series = np.append(reward_series, avg_rewards) # flatten
         switching_series = np.append(switching_series, switching) # flatten
         collision_series = np.append(collision_series, collisions) # flatten
         alg_indicator_series.extend([simulation_results['algorithm_name'][alg_id]] * simu_rounds)
         network_size_indicator_series.extend(network_sizes)
       
                
    prepared_results = {}                
    prepared_results['Sum of rewards'] = reward_series            
    prepared_results['Node Number'] = network_size_indicator_series     
    prepared_results['Accumulated switching counts'] = switching_series
    prepared_results['Accumulated collision counts'] = collision_series
    prepared_results['Algorithms'] = alg_indicator_series
    
#    print("length: {}, {}, {}, {}, {}".format(len(reward_series), len(network_size_indicator_series),
#          len(switching_series), len(collision_series), len(alg_indicator_series)))
            
    simu_data_frame = pd.DataFrame(prepared_results)
            
    
    return simu_data_frame
            

def simulation_plot_results(input_data_frame):
    #plot and save the figure: 1
    file_name = "network_switching"
    plot_data_frame(input_data_frame, 
                    xlabel="Node Number", ylabel="Accumulated switching counts", huelabel='Algorithms', 
                    flag_semilogx = False,
                    save_file_name=file_name, save_data_name=None)
            
    #plot and save the figure: 2
    file_name = "network_collision"
    plot_data_frame(input_data_frame, 
                    xlabel="Node Number", ylabel="Accumulated collision counts", huelabel='Algorithms', 
                    flag_semilogx = False,
                    save_file_name=file_name, save_data_name=None)          
    
    file_name = "network_rewards"
    plot_data_frame(input_data_frame, 
                    xlabel="Node Number", ylabel="Sum of rewards", huelabel='Algorithms', 
                    flag_semilogx = False,
                    save_file_name=file_name, save_data_name=None)

if __name__ == '__main__':    
    """
    Parallel processing is turned off by default. 
    Unless the machine memory is sufficiently large, we may have a risk of running out of memory 
    for a large network scale.
    """
    yes = {'yes','y', 'ye', 'Y'}
    no = {'no','n', 'N'}
    
    print("This simulation takes more than 10 hrs. \nDo you want to continue? [y/n]")
    while True:
        input_choice = input().lower()
        if input_choice in yes:
           break
        elif input_choice in no:
           print('execution is terminated.')
           sys.exit()
        else:
           print("Please respond with 'yes' or 'no'")
    
    game_horizon = 400000
    simu_rounds = 40
  
    max_player_no = 30 # the more nodes we have, the longer horizon we need for find a social-optimal allocation.
    
    player_numbers = np.linspace(5, max_player_no, 6)  #Example: [max_player_no, 25, 20, 15, 10, 5]
    max_arm_number = max_player_no + 1 # to save some memory
    
    env_config = {'horizon': game_horizon,
                  'arm number': max_arm_number,
                  'player number': max_player_no,
                  'context set': {"context 1", "context 2", "context 3"},#
                  'env_type': 'HetNet simulator', # change the underlying distribution here
                  'enabel mmWave': True,
                  'cell range': 250,
                  'context_prob': {'context 1': 2, 'context 2': 1, 'context 3': 1},
                  'los_prob':  {'context 1': 1.5, 'context 2': 2, 'context 3': 1}
                  }
        
    # generate the arm-value sequence for only once
    alg_engine = AlgEvaluator(env_config)  
    alg_engine.prepare_arm_samples()
        
    game_config = Section("Simulation of HetNet: reward evolution for 4 algorithms")    
    game_config.alg_types = ['Musical Chairs', 'SOC', 'Trial and Error', 'Game of Thrones'] #, 
  
    # beginning of the game
    start_time = time.time()# record the starting time of the simulation, start simulations
    data_frame = []
    for player_no in player_numbers:        
        num_players = int(player_no)
        
        # be sure that the value of the two constant variables satisfiy the condition in Theorem 2 of [Wang2020]
        alpha11 = -0.40/num_players
        alpha12 = 0.45/num_players
        
        game_config.alg_configs = [None,
                                  {"delta": 0.02, "exploration_time": 4000},
                                  {"c1": 2000, "c2": 10000,"c3":3000, "epsilon": 0.01, "delta": 2, "xi": 0.001, 
                                                     "alpha11": alpha11, "alpha12": alpha12, "alpha21": -0.39, "alpha22": 0.4,},
                                  {"c1": 2000, "c2": 10000,"c3":3000, "epsilon": 0.01, "delta": 1.5},                                 
                                  ]  
        
        #set the arm number to be used in the simulation
        alg_engine.reset_player_number(num_players)
        alg_engine.reset_arm_number(num_players + 1)   
        alg_engine.clear_algorithms()
        
        if player_no >= 10:
            temp_simu_data_frame = simulation_execution(alg_engine, game_config, num_players, game_horizon, simu_rounds) 
        else:
            # There is always a risk of overwhelming the memory capacity with parallel processing, especially when num_players > 15 
            # Set the last parameter to True to enable parallel processing
            temp_simu_data_frame = simulation_execution(alg_engine, game_config, num_players, game_horizon, simu_rounds, False) 

        data_frame.append(temp_simu_data_frame)

    #end of the numerical simulation        
    input_data = pd.concat(data_frame)        
    running_time = time.time() - start_time    
    print("Simulation completes in {}.".format(datetime.timedelta(seconds=running_time)))

    #plotting figures
    simulation_plot_results(input_data)