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
# This file is the main entrance of all the simulations except that for those w.r.t. network sizes.


__author__ = "Wenbo Wang"

import numpy as np
import pandas as pd

import time
import datetime
import argparse

from GameEvaluator import AlgEvaluator
from plotutils import plot_data_frame, plot_repeated_simu_results

import simu_config as CONFIG

def simulation_execution(game_config):        
    """
    simulation_execution() is the main body of the MP-MAP algorithm simulations
    """    
    print("MAB game with configuration '{}' starts to play...".format(game_config.__repr__()))

    game_horizon = game_config.game_horizon
    alg_engine = AlgEvaluator(game_config.env_config)   

    #add algorithms
    for alg_id in range(len(game_config.alg_types)):
        alg_engine.add_algorithm(algo_type=game_config.alg_types[alg_id], 
                                 custome_params=game_config.alg_configs[alg_id])
    
    print("MAB game prepares the environment for arm type '{}' of {} rounds".format(game_config.env_config['env_type'], game_horizon))
    alg_engine.prepare_arm_samples()
    
    # simulation 1: reward plotting to compare the efficiency of the algorithms
    if "enable_efficiency_simulation" in game_config.__dict__ and game_config.enable_efficiency_simulation:
        start_time_oneshot = time.time()
                
        #######################################################################
        #
        if game_config.flag_parallel != True:
#            print("starting single-process simulation...")
            alg_engine.play_game(flag_progress_bar=game_config.flag_progress_bar)    
        else:
#            print("starting parallel simulation...")
            alg_engine.play_game_parallel(flag_progress_bar=game_config.flag_progress_bar)
        #
        #######################################################################
            
        alg_engine.plot_rewards(save_fig = game_config.flag_save_figure, save_data = game_config.save_data)        
        
        # printing
        running_time = time.time() - start_time_oneshot           
        print("Single-shot simulation completes in {} for {} iterations.".format( \
                datetime.timedelta(seconds=running_time), game_horizon))
    
    # simulation 2/3/4: plotting regret or total rewards over horizon
    if ("enable_regret_simulation" in game_config.__dict__ and game_config.enable_regret_simulation) or \
       ("enable_reward_simulation" in game_config.__dict__ and game_config.enable_reward_simulation) or \
       ("enable_switching_simulation" in game_config.__dict__ and game_config.enable_switching_simulation):    
        start = game_config.T_start
        nb_point = game_config.T_step
        
        horizon_list = np.exp(np.linspace(np.log(start), np.log(game_horizon), nb_point))
        simu_rounds = game_config.T_simu_rounds
        
        start_time_repeated = time.time()
        
        #######################################################################
        #
        if game_config.flag_parallel != True:
#            print("starting single-process simulation...")
            simulation_results = alg_engine.play_repeated_game(horizon_list, simulation_rounds=simu_rounds, 
                                                               flag_progress_bar=game_config.flag_progress_bar)                        
        else:
#            print("starting parallel simulation...")
            simulation_results = alg_engine.play_repeated_game_parallel(horizon_list, simulation_rounds=simu_rounds,
                                                                        flag_progress_bar=game_config.flag_progress_bar)
        #
        #######################################################################
          
        # printing
        running_time = time.time() - start_time_repeated    
        print("Repeated simulation completes in {} with maximum horizon {} in {} rounds of plays...".format(\
              datetime.timedelta(seconds=running_time), game_horizon, simu_rounds))
            
        # virtualization for simulation 2
        if "enable_regret_simulation" in game_config.__dict__ and game_config.enable_regret_simulation:
            # locate the reference algorithm
            optimal_alg_id = 0
            
            len_horizon = simulation_results['horizon'].shape[1]
            time_series = np.empty((0, len_horizon))
            alg_indicator_series = []
    
            avg_regret_series = np.empty((0, len_horizon))
            for alg_id in range(len(simulation_results['algorithm_name'])):
                if alg_id != optimal_alg_id:
                    # the returned value simulation_results['reward_series'] is organized as an array:
                    # (len(algorithm_ids), simulation_rounds*len(horizon_list))
                    horizon_series = simulation_results['horizon'][alg_id,:]
                    avg_regret = (simulation_results['reward_series'][optimal_alg_id,:] - 
                              simulation_results['reward_series'][alg_id,:]) / horizon_series
                
                    avg_regret_series = np.append(avg_regret_series, avg_regret) # flatten
                    time_series = np.append(time_series, horizon_series)
                    
                    alg_indicator_series.extend([simulation_results['algorithm_name'][alg_id]] * len(horizon_series))
                    
            prepared_results = {}                
            prepared_results['Average regret'] = avg_regret_series            
            prepared_results['Total number of plays'] = time_series        
            prepared_results['Algorithms'] = alg_indicator_series
                
            simu_data_frame = pd.DataFrame(prepared_results)
            
            # plot and save the figure    
            file_name = "monte_carlo_regret" if game_config.flag_save_figure==True else None        
            sns_figure_unused, repeated_play_data_name = plot_data_frame(simu_data_frame, 
                            xlabel="Total number of plays", ylabel="Average regret", huelabel='Algorithms', 
                            save_file_name=file_name, save_data_name=game_config.repeated_play_data_name)
            
            # post processing, add the theoretical bound to the figure
            flag_bound = False
            if hasattr(game_config, 'flag_regret_bound'):
                flag_bound = game_config.flag_regret_bound
            else:
                flag_bound = False
                
            plot_repeated_simu_results(start=start, horzion=game_horizon, nbPoints=nb_point, flag_bound=flag_bound,
                                       data_file_name=repeated_play_data_name)            
        
        # virtualization for simulation 3
        if "enable_reward_simulation" in game_config.__dict__ and game_config.enable_reward_simulation:          
            len_horizon = simulation_results['horizon'].shape[1]
            time_series = np.empty((0, len_horizon))
            alg_indicator_series = []
            
            reward_series = np.array([])
            for alg_id in range(len(simulation_results['algorithm_name'])):
                horizon_series = simulation_results['horizon'][alg_id,:]
                avg_rewards = simulation_results['reward_series'][alg_id, :] / horizon_series
                
                reward_series = np.append(reward_series, avg_rewards) # flatten
                time_series = np.append(time_series, horizon_series)
                alg_indicator_series.extend([simulation_results['algorithm_name'][alg_id]] * len(horizon_series))
                
            prepared_results = {}                
            prepared_results['Average sum of rewards'] = reward_series            
            prepared_results['Total number of plays'] = time_series        
            prepared_results['Algorithms'] = alg_indicator_series
            
            simu_data_frame = pd.DataFrame(prepared_results)
            
            #plot and save the figure    
            file_name = "monte_carlo_rewards" if game_config.flag_save_figure==True else None        
            plot_data_frame(simu_data_frame, 
                            xlabel="Total number of plays", ylabel="Average sum of rewards", huelabel='Algorithms', 
                            flag_semilogx = False,
                            save_file_name=file_name, save_data_name=game_config.repeated_play_data_name)
            
        # virtualization for simulation 4    
        if "enable_switching_simulation" in game_config.__dict__ and game_config.enable_switching_simulation:
            len_horizon = simulation_results['horizon'].shape[1]
            time_series = np.empty((0, len_horizon))
            alg_indicator_series = []
            
            switching_series = np.array([])
            collision_series = np.array([])
                       
            for alg_id in range(len(simulation_results['algorithm_name'])):
                horizon_series = simulation_results['horizon'][alg_id,:]
                switching = simulation_results['switching_count_series'][alg_id, :]
                collisions = simulation_results['collision_series'][alg_id, :]
                
                switching_series = np.append(switching_series, switching) # flatten
                collision_series = np.append(collision_series, collisions) # flatten
                
                time_series = np.append(time_series, horizon_series)
                alg_indicator_series.extend([simulation_results['algorithm_name'][alg_id]] * len(horizon_series))
                
            prepared_results = {}                
            prepared_results['Accumulated switching counts'] = switching_series
            prepared_results['Accumulated collision counts'] = collision_series
            prepared_results['Total number of plays'] = time_series        
            prepared_results['Algorithms'] = alg_indicator_series
            
            assert len(switching_series) == len(collision_series), "switching array must be of the same length: {}, {}".format(
                    len(switching_series), len(collision_series))
            
            simu_data_frame = pd.DataFrame(prepared_results)
            
            #plot and save the figure: 1
            file_name = "monte_carlo_switching" if game_config.flag_save_figure==True else None        
            plot_data_frame(simu_data_frame, 
                            xlabel="Total number of plays", ylabel="Accumulated switching counts", huelabel='Algorithms', 
                            flag_semilogx = False,
                            save_file_name=file_name, save_data_name=game_config.repeated_play_data_name)
            
            #plot and save the figure: 2
            file_name = "monte_carlo_collision" if game_config.flag_save_figure==True else None        
            plot_data_frame(simu_data_frame, 
                            xlabel="Total number of plays", ylabel="Accumulated collision counts", huelabel='Algorithms', 
                            flag_semilogx = False,
                            save_file_name=file_name, save_data_name=game_config.repeated_play_data_name)            


if __name__ == '__main__':    
    """
    Parallel processing is suggested to be turned on for repeated simulations (see simu_config.py)
    It is approximately 2X to 4X faster in terms of the total time than the single-process simulation
    """
    arg_parser = argparse.ArgumentParser(description='Select a configuration set in \'simu_config.py\' to run the simulations')    
    # Add the arguments
    arg_parser.add_argument('-id',  metavar='ID', type=int,
                           help='Choose the configuration ID between [1-13], see the summary of simu_config.py')
    args = arg_parser.parse_args()  

    if args.id is None:
        # default choice of configuration for a simulation
        game_config = CONFIG.ENV_SCENARIO_7 # 
    else:
        if args.id in CONFIG.CONFIGURATION_DICT.keys():
            game_config =  CONFIG.CONFIGURATION_DICT[args.id]
        else:
            raise Exception('the input configuration ID is not valid')      
    
    # beginning of the game
    start_time = time.time()# record the starting time of the simulation, start simulations

    simulation_execution(game_config)
        
    #end of the game
    running_time = time.time() - start_time    
    print("Simulation completes in {}.".format(datetime.timedelta(seconds=running_time)))
