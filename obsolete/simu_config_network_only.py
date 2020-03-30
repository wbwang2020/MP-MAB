# -*- coding: utf-8 -*-
"""
@author: Wenbo Wang

License:
This program is licensed under the GPLv2 license. If you in any way use this code for research 
that results in publications, please cite our original article listed above.
 
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
See the GNU General Public License for more details.
"""

# This file provides the configurations for each simulation.

__author__ = "Wenbo Wang"

from envutils import Struct as Section

if __name__ == '__main__':
    print("Warning: this script 'simu_config.py' is NOT executable..")  # DEBUG
    exit(0)

###############################################################################
# Section 1:
# Define the algorithms that are used in the simulation
###############################################################################
ENV_ALG_SETTING_1 = Section("Simulation of HetNet: reward evolution for 5 algorithms")
ENV_ALG_SETTING_1.game_horizon = 200000


# Disable simulation for reward evolution in a single shot
ENV_ALG_SETTING_1.enable_reward_simulation = True
ENV_ALG_SETTING_1.enable_switching_simulation = True

ENV_ALG_SETTING_1.alg_types = ['Musical Chairs', 'SOC', 'Trial and Error', 'Game of Throne', 'TnE Nonobservable'] #, 
ENV_ALG_SETTING_1.alg_configs = [None,                               
                              {"delta": 0.02, "exploration_time": 4000},
                              {"c1": 100, "c2": 200,"c3":100, "epsilon": 0.01, "delta": 2, "xi": 0.001, 
                                                 "alpha11": -0.12, "alpha12": 0.15, "alpha21": -0.39, "alpha22": 0.4,},
                              {"c1": 100, "c2": 300,"c3":200, "epsilon": 0.025, "delta": 1.5},
                              {"c1": 100, "c2": 200,"c3":100, "epsilon": 0.025, "delta": 1.5, "xi": 0.001, 
                                                 "alpha11": -0.12, "alpha12": 0.15, "alpha21": -0.35, "alpha22": 0.4, "observable": 0}
                              ]
                             
# Experiment parameters
ENV_ALG_SETTING_1.flag_save_figure = True
ENV_ALG_SETTING_1.save_data = False

# Experiment parameters
ENV_ALG_SETTING_1.T_repr_rounds = 40

ENV_ALG_SETTING_1.repeated_play_data_name = 'reward_data_4_alg_HetNet'

# Enable parallel processing
ENV_ALG_SETTING_1.flag_parallel = True
ENV_ALG_SETTING_1.flag_progress_bar = True