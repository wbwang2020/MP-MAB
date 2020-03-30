# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:28:25 2020

@author: wenbo2017
"""

# remeber to change the names of data source files

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import pandas as pd
from plotutils import plot_data_frame, plot_repeated_simu_results

data_reward = pd.read_pickle('reward_data_4_alg_HetNet--2020-03-27-11-22-00.pkl')

plot_data_frame(data_reward, 
                xlabel="Total number of plays", ylabel="Average sum of rewards", huelabel='Algorithms', 
                flag_semilogx = False,
                save_file_name=None, save_data_name=None)


data_reward = pd.read_pickle('reward_data_4_alg_HetNet--2020-03-27-11-22-03.pkl')

plot_data_frame(data_reward, 
                xlabel="Total number of plays", ylabel="Accumulated switching counts", huelabel='Algorithms', 
                flag_semilogx = False,
                save_file_name=None, save_data_name=None)

data_reward = pd.read_pickle('reward_data_4_alg_HetNet--2020-03-27-11-22-03.pkl')

plot_data_frame(data_reward, 
                xlabel="Total number of plays", ylabel="Accumulated collision counts", huelabel='Algorithms', 
                flag_semilogx = False,
                save_file_name=None, save_data_name=None)