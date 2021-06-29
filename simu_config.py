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

# This file provides the configurations for each simulation.

__author__ = "Wenbo Wang"

import numpy as np
from envutils import Struct as Section

if __name__ == '__main__':
    print("Warning: this script 'simu_config.py' is NOT executable..")  # DEBUG
    exit(0)

# (context-player): arm-vector: {lower bound} - {upper bound}
initial_data =  [{("context 1", 0): np.array([0., 0.5, 0.3]), ("context 2", 0): np.array([0.1, 0.2, 0.2]), ("context 3", 0): np.array([0., 0.2, 0.25]),
                 ("context 1", 1): np.array([0.1, 0.6, 0.2]), ("context 2", 1): np.array([0., 0., 0.]), ("context 3", 1): np.array([0.2, 0.1, 0.45])},
                {("context 1", 0): np.array([0.5, 0.8, 0.6]), ("context 2", 0): np.array([1., 1., 0.4]), ("context 3", 0): np.array([1, 0.3, 0.65]),
                 ("context 1", 1): np.array([0.81, 0.96, 0.52]), ("context 2", 1): np.array([0.5, 0.4, 0.9]), ("context 3", 1): np.array([0.62, 0.21, 0.95])}    
                ]

initial_data_2 =  [{("context 1", 0): np.array([0.0, 0.5, 0.3, 0.1]), ("context 2", 0): np.array([0.1, 0.2, 0.2, 0.5]), ("context 3", 0): np.array([0.0, 0.2, 0.25, 0.4]),
                    ("context 1", 1): np.array([0.1 , 0.6 , 0.2 , 0.44]), ("context 2", 1): np.array([0.0, 0.0, 0.0, 0.2]), ("context 3", 1): np.array([0.2 , 0.1 , 0.45, 0.36]),
                    ("context 1", 2): np.array([0.24, 0.11, 0.3 , 0.14]), ("context 2", 2): np.array([0.2, 0.0 , 0.1, 0.2]), ("context 3", 2): np.array([0.32, 0.21, 0.25, 0.59])},

                   {("context 1", 0): np.array([0.5, 0.8, 0.6, 0.7]), ("context 2", 0): np.array([1.0, 1.0, 0.4, 1.0]), ("context 3", 0): np.array([1.0, 0.3, 0.65, 0.9]),
                    ("context 1", 1): np.array([0.81, 0.96, 0.52, 1.0 ]), ("context 2", 1): np.array([0.5, 0.4, 0.9, 0.6]), ("context 3", 1): np.array([0.62, 0.31, 0.95, 0.79]),
                    ("context 1", 2): np.array([0.81, 0.78, 0.67, 1.0 ]), ("context 2", 2): np.array([0.3, 0.95, 0.9, 0.6]), ("context 3", 2): np.array([0.75, 0.63, 1.0 , 0.99]),}    
                ]

###############################################################################
# Section 1:
# Hard-coded MAB environment for uniform/gaussian arms and unifrom context with
# 3-contexts, 2-plaers, 3-arms
###############################################################################
ENV_SCENARIO_1 = Section("2-player-3-context-3-unifroms-arm MAB: regret evolution")
ENV_SCENARIO_1.game_horizon = 200000
ENV_SCENARIO_1.env_config = {'horizon': ENV_SCENARIO_1.game_horizon,
                      'arm number': 3,
                      'player number': 2,
                      'context set': {"context 1", "context 2", "context 3"},#
                      'env_type': 'uniform', # change the underlying distribution here
                      'initial data': initial_data
                      }
# Enable simulation for regret evolution with repetition
ENV_SCENARIO_1.enable_regret_simulation = True
ENV_SCENARIO_1.enable_reward_simulation = True

ENV_SCENARIO_1.alg_types = ['Static Hungarian', 'Musical Chairs', 'ESE', 'SIC-MMAB', 'Trial and Error', 'Game of Thrones'] # 
ENV_SCENARIO_1.alg_configs = [None, None, None, None, {"c1": 500, "c2": 1000,"c3":1000, "epsilon": 0.01, "delta": 2, "xi": 0.001, 
                                                 "alpha11": -0.12, "alpha12": 0.15, "alpha21": -0.35, "alpha22": 0.4,},
                              {"c1": 500, "c2": 1000,"c3":1000, "epsilon": 0.01, "delta": 2}]
                             
# Experiment parameters
ENV_SCENARIO_1.T_start = 5000
ENV_SCENARIO_1.T_step = 20
ENV_SCENARIO_1.T_simu_rounds = 20

ENV_SCENARIO_1.flag_save_figure = True
ENV_SCENARIO_1.repeated_play_data_name = 'regret_data'

# Enable parallel processing
ENV_SCENARIO_1.flag_parallel = False
ENV_SCENARIO_1.flag_progress_bar = True

###############################################################################
# Section 1:
# Parallel version
###############################################################################
ENV_SCENARIO_1_PARALLEL = ENV_SCENARIO_1
ENV_SCENARIO_1_PARALLEL.flag_parallel = True

###############################################################################
# Section 2:
# Hard-coded MAB environment for uniform/gaussian arms and unifrom context with
# 3-contexts, 2-plaers, 3-arms
###############################################################################
ENV_SCENARIO_2 = Section("2-player-3-context-3-unifroms-arm MAB: reward evolution")
ENV_SCENARIO_2.game_horizon = 80000
ENV_SCENARIO_2.env_config = {'horizon': ENV_SCENARIO_2.game_horizon,
                      'arm number': 3,
                      'player number': 2,
                      'context set': {"context 1", "context 2", "context 3"},#
                      'env_type': 'uniform', # change the underlying distribution here
                      'initial data': initial_data
                      }

# Disable simulation for reward evolution in a single shot
ENV_SCENARIO_2.enable_efficiency_simulation = True

ENV_SCENARIO_2.alg_types = ['Static Hungarian', 'Musical Chairs', 'Trial and Error']
ENV_SCENARIO_2.alg_configs = [None, None, {"c1": 100, "c2": 200,"c3":100, "epsilon": 0.01, "delta": 2, "xi": 0.001, 
                                                 "alpha11": -0.12, "alpha12": 0.15, "alpha21": -0.35, "alpha22": 0.4,}]
                             
# Experiment parameters
ENV_SCENARIO_2.flag_save_figure = True
ENV_SCENARIO_2.save_data = True

# Enable parallel processing
ENV_SCENARIO_2.flag_parallel = False
ENV_SCENARIO_2.flag_progress_bar = True

###############################################################################
# Section 2:
# Parallel version
###############################################################################
ENV_SCENARIO_2_PARALLEL = ENV_SCENARIO_2
ENV_SCENARIO_2_PARALLEL.flag_parallel = True

###############################################################################
# Section 3:
# Hard-coded MAB environment for uniform/gaussian arms and unifrom context with
# 3-contexts, 2-plaers, 3-arms
###############################################################################
ENV_SCENARIO_3 = Section("2-player-3-context-3-unifroms-arm MAB: regret evolution")
ENV_SCENARIO_3.game_horizon = 200000
ENV_SCENARIO_3.env_config = {'horizon': ENV_SCENARIO_3.game_horizon,
                      'arm number': 3,
                      'player number': 2,
                      'context set': {"context 1", "context 2", "context 3"},#
                      'env_type': 'uniform', # change the underlying distribution here
                      'initial data': initial_data
                      }

# Enable simulation for regret evolution with repetition
ENV_SCENARIO_3.enable_regret_simulation = True

ENV_SCENARIO_3.alg_types = ['Static Hungarian', 'Musical Chairs', 'Trial and Error', 'Game of Thrones']
ENV_SCENARIO_3.alg_configs = [None, None, {"c1": 100, "c2": 200,"c3":100, "epsilon": 0.01, "delta": 2, "xi": 0.001, 
                                                 "alpha11": -0.12, "alpha12": 0.15, "alpha21": -0.35, "alpha22": 0.4,}, 
                           {"c1": 100, "c2": 200,"c3":100, "epsilon": 0.01, "delta": 2}]
                             
# Experiment parameters
ENV_SCENARIO_3.flag_save_figure = True
ENV_SCENARIO_3.save_data = True

# Experiment parameters
ENV_SCENARIO_3.T_start = 5000
ENV_SCENARIO_3.T_step = 20
ENV_SCENARIO_3.T_simu_rounds = 200

ENV_SCENARIO_3.repeated_play_data_name = 'regret_data_3_alg'

# Enable parallel processing
ENV_SCENARIO_3.flag_parallel = False
ENV_SCENARIO_3.flag_progress_bar = True

###############################################################################
# Section 3:
# Parallel version
###############################################################################
ENV_SCENARIO_3_PARALLEL = ENV_SCENARIO_3
ENV_SCENARIO_3_PARALLEL.flag_parallel = True

###############################################################################
# Section 4:
# Hard-coded MAB environment for uniform/gaussian arms and unifrom context with
# 3-contexts, 2-plaers, 3-arms, test of parallel simulation
# for a single round of this 4-algorithm example, multiprocessing accelerates by
# about 1/3
###############################################################################
ENV_SCENARIO_4 = Section("2-player-3-context-3-unifroms-arm MAB: reward evolution")
ENV_SCENARIO_4.game_horizon = 200000
ENV_SCENARIO_4.env_config = {'horizon': ENV_SCENARIO_4.game_horizon,
                      'arm number': 3,
                      'player number': 2,
                      'context set': {"context 1", "context 2", "context 3"},#
                      'env_type': 'uniform', # change the underlying distribution here
                      'initial data': initial_data
                      }

# Disable simulation for reward evolution in a single shot
ENV_SCENARIO_4.enable_efficiency_simulation = True

ENV_SCENARIO_4.alg_types = ['Static Hungarian', 'Musical Chairs', 'Trial and Error', 'Game of Thrones']
ENV_SCENARIO_4.alg_configs = [None, None, {"c1": 100, "c2": 200,"c3":100, "epsilon": 0.01, "delta": 2, "xi": 0.001, 
                                                 "alpha11": -0.12, "alpha12": 0.15, "alpha21": -0.35, "alpha22": 0.4,},
                       {"c1": 100, "c2": 200,"c3":100, "epsilon": 0.01, "delta": 2}]
                             
# Experiment parameters
ENV_SCENARIO_4.flag_save_figure = True
ENV_SCENARIO_4.save_data = False

# Experiment parameters
ENV_SCENARIO_4.T_start = 5000
ENV_SCENARIO_4.T_step = 20
ENV_SCENARIO_4.T_simu_rounds = 200

ENV_SCENARIO_4.repeated_play_data_name = 'regret_data_3_alg'

# Enable parallel processing
ENV_SCENARIO_4.flag_parallel = False
ENV_SCENARIO_4.flag_progress_bar = True

###############################################################################
# Section 4:
# Parallel version
###############################################################################
ENV_SCENARIO_4_PARALLEL = ENV_SCENARIO_4
ENV_SCENARIO_4_PARALLEL.flag_parallel = True

###############################################################################
# Section 5:
# MAB environment in HetNet, with 12 random arms/channel and 10 randomly placed
# users, 3 contexts (MUE transmission in the underlying macro cells)
# for a single round of this 4-algorithm example, multiprocessing is to be implemented
###############################################################################
ENV_SCENARIO_5 = Section("10-UE-10-Channel HetNet: regret evolution")
ENV_SCENARIO_5.game_horizon = 80000
ENV_SCENARIO_5.env_config = {'horizon': ENV_SCENARIO_5.game_horizon,
                      'arm number': 12,
                      'player number': 10,
                      'context set': {"context 1", "context 2", "context 3"},#
                      'env_type': 'HetNet simulator', # change the underlying distribution here
                      'enabel mmWave': True,
                      'cell range': 200,
                      'context_prob': {'context 1': 1, 'context 2': 1, 'context 3': 1},
                      'los_prob':  {'context 1': 1, 'context 2': 1, 'context 3': 1}
                      }

# Disable simulation for reward evolution in a single shot
ENV_SCENARIO_5.enable_efficiency_simulation = True

ENV_SCENARIO_5.alg_types = ['Musical Chairs', 'Trial and Error', 'Game of Thrones']
ENV_SCENARIO_5.alg_configs = [None, {"c1": 100, "c2": 200,"c3":100, "epsilon": 0.01, "delta": 2, "xi": 0.001, 
                                                 "alpha11": -0.12, "alpha12": 0.15, "alpha21": -0.39, "alpha22": 0.4,},
                       {"c1": 100, "c2": 200,"c3":100, "epsilon": 0.01, "delta": 2}]
                             
# Experiment parameters
ENV_SCENARIO_5.flag_save_figure = True
ENV_SCENARIO_5.save_data = False

# Experiment parameters
ENV_SCENARIO_5.T_start = 5000
ENV_SCENARIO_5.T_step = 20
ENV_SCENARIO_5.T_simu_rounds = 200

ENV_SCENARIO_5.repeated_play_data_name = 'regret_data_3_alg'

# Enable parallel processing
ENV_SCENARIO_5.flag_parallel = False
ENV_SCENARIO_5.flag_progress_bar = True

###############################################################################
# Section 5:
# Parallel version
###############################################################################
ENV_SCENARIO_5_PARALLEL = ENV_SCENARIO_5
ENV_SCENARIO_5_PARALLEL.flag_parallel = True

###############################################################################
# Section 6:
# MAB environment in HetNet, with 12 random arms/channel and 10 randomly placed
# users, 3 contexts (MUE transmission in the underlying macro cells)
###############################################################################
ENV_SCENARIO_6 = Section("10-UE-12-Channel HetNet: reward evolution")
ENV_SCENARIO_6.game_horizon = 200000
ENV_SCENARIO_6.env_config = {'horizon': ENV_SCENARIO_6.game_horizon,
                      'arm number': 12,
                      'player number': 10,
                      'context set': {"context 1", "context 2", "context 3"},#
                      'env_type': 'HetNet simulator', # change the underlying distribution here
                      'enabel mmWave': True,
                      'cell range': 250,
                      'context_prob': {'context 1': 2, 'context 2': 1, 'context 3': 1},
                      'los_prob':  {'context 1': 1.5, 'context 2': 2, 'context 3': 1}
                      }

# Disable simulation for reward evolution in a single shot
ENV_SCENARIO_6.enable_efficiency_simulation = False
ENV_SCENARIO_6.enable_regret_simulation = False
ENV_SCENARIO_6.enable_reward_simulation = True
ENV_SCENARIO_6.enable_switching_simulation = True

ENV_SCENARIO_6.alg_types = ['Musical Chairs', 'SOC', 'Trial and Error', 'Game of Thrones'] #, 
ENV_SCENARIO_6.alg_configs = [None,                               
                              {"delta": 0.02, "exploration_time": 10000},
                              {"c1": 1000, "c2": 3000,"c3":3000, "epsilon": 0.01, "delta": 1.5, "xi": 0.001, 
                                                 "alpha11": -0.04, "alpha12": 0.05, "alpha21": -0.035, "alpha22": 0.04, "observable": 1},
                              {"c1": 1000, "c2": 3000,"c3":3000, "epsilon": 0.01, "delta": 1.5},
                              ]
                             
# Experiment parameters
ENV_SCENARIO_6.flag_save_figure = True
ENV_SCENARIO_6.save_data = False

# Experiment parameters
ENV_SCENARIO_6.T_start = 40000
ENV_SCENARIO_6.T_step = 12
ENV_SCENARIO_6.T_simu_rounds = 200

ENV_SCENARIO_6.repeated_play_data_name = 'reward_data_4_alg_HetNet'

# Enable parallel processing
ENV_SCENARIO_6.flag_parallel = False
ENV_SCENARIO_6.flag_progress_bar = True

###############################################################################
# Section 6:
# Parallel version
###############################################################################
ENV_SCENARIO_6_PARALLEL = ENV_SCENARIO_6
ENV_SCENARIO_6_PARALLEL.flag_parallel = True

###############################################################################
# Section 7:
# Hard-coded MAB environment for uniform/gaussian arms and unifrom context with
# 3-contexts, 2-plaers, 3-arms
###############################################################################
ENV_SCENARIO_7 = Section("3-context-3-player-4-unifroms-arm MAB: reward evolution")
ENV_SCENARIO_7.game_horizon = 100000
ENV_SCENARIO_7.env_config = {'horizon': ENV_SCENARIO_7.game_horizon,
                      'arm number': 4,
                      'player number': 3,
                      'context set': {"context 1", "context 2", "context 3"},#
                      'env_type': 'uniform', # change the underlying distribution here
                      'initial data': initial_data_2
                      }

# add algorithms
ENV_SCENARIO_7.alg_types = ['Musical Chairs', 'SOC', 'Game of Thrones', 'Trial and Error'] #,  , 'TnE Nonobservable'
ENV_SCENARIO_7.alg_configs = [None, 
                              {"delta": 0.02, "exploration_time": 10000},
                              {"c1": 500, "c2": 1000,"c3":1000, "epsilon": 0.01, "delta": 1.5}, 
                              {"c1": 500, "c2": 1000,"c3":1000, "epsilon": 0.01, "delta": 1.5, "xi": 0.001, 
                                                 "alpha11": -0.12, "alpha12": 0.15, "alpha21": -0.35, "alpha22": 0.4},
#                              {"c1": 300, "c2": 1000,"c3":1000, "epsilon": 0.01, "delta": 1.5, "xi": 0.001, 
#                                                 "alpha11": -0.12, "alpha12": 0.15, "alpha21": -0.35, "alpha22": 0.4, "observable": 0}
                               ]

# Disable simulation for reward evolution in a single shot
ENV_SCENARIO_7.enable_efficiency_simulation = False
ENV_SCENARIO_7.enable_regret_simulation = False
ENV_SCENARIO_7.enable_reward_simulation = True
ENV_SCENARIO_7.enable_switching_simulation = True

# Experiment parameters
ENV_SCENARIO_7.T_start = 20000
ENV_SCENARIO_7.T_step = 10
ENV_SCENARIO_7.T_simu_rounds = 20

ENV_SCENARIO_7.repeated_play_data_name = 'congfig_7_5_algs_uniform'
        
# Experiment parameters
ENV_SCENARIO_7.flag_save_figure = True
ENV_SCENARIO_7.save_data = False

# Enable parallel processing
ENV_SCENARIO_7.flag_parallel = True
ENV_SCENARIO_7.flag_progress_bar = True

###############################################################################
# All configurations are stored in the following dictionary:
###############################################################################
CONFIGURATION_DICT = {1: ENV_SCENARIO_1,
                      2: ENV_SCENARIO_2,
                      3: ENV_SCENARIO_3,
                      4: ENV_SCENARIO_4,
                      5: ENV_SCENARIO_5,
                      6: ENV_SCENARIO_6,       
                      7: ENV_SCENARIO_1_PARALLEL,   
                      8: ENV_SCENARIO_2_PARALLEL,
                      9: ENV_SCENARIO_3_PARALLEL,
                      10: ENV_SCENARIO_4_PARALLEL,
                      11: ENV_SCENARIO_5_PARALLEL,
                      12: ENV_SCENARIO_6_PARALLEL,
                      13: ENV_SCENARIO_7
        }
