# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:06:46 2019

@author: wenbo2017
"""

# testing plotting methods in 
import seaborn as sns
import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt
from plotutils import plot_data_frame, prepare_file_name, read_data_frame, plot_repeated_simu_results
#from matplotlib.lines import Line2D
import simu_config as CONFIG

flag_test1 = False            
 # start the simulation    
if flag_test1 == True:
    horizon_list = np.linspace(5000, 50000, 20)
  
    timepoints = []
    alg_len = 3
    regret_series = []
    #Monte Carlo Simulation
    simu_rounds = 300
    
    alg_types = [ii for ii in range(alg_len)]
    
    type_series = []

    for simu_index in range(simu_rounds):
        print("Simulation round {} of total rounds {}...".format(simu_index, simu_rounds))
        # 2d array of payoff for a single simulation round
        learned_total_payoff = np.zeros((alg_len, len(horizon_list)))
        
        horizon_index = 0
        for horizon_index in range(len(horizon_list)):            
            # example: for 3 algorithms, len(tmp_total_payoff) == 3
            tmp_total_payoff = np.random.rand(alg_len)
            
            for alg_index in range(alg_len):
                learned_total_payoff[alg_index][horizon_index] = tmp_total_payoff[alg_index] #/ horizon_list[horizon_index]
                
            type_series.extend(alg_types)
            
            tmp_time = [horizon_list[horizon_index]]*alg_len
            
            timepoints.extend(tmp_time)
            
            regret_series.extend(tmp_total_payoff)
        
    recorded_data = {}
            
    recorded_data['signal'] = regret_series
        
    recorded_data['time'] = timepoints
    
    recorded_data['algorithms'] = type_series
    
    my_data = pd.DataFrame(recorded_data)
    
#    sns.relplot(x="time", y="signal", hue = 'algorithms',
#            kind="line", data=my_data, height=5, aspect=1.25 );
    plot_data_frame(my_data, xlabel="time", ylabel="signal", huelabel='algorithms', save_file_name='test')

flag_test2=False
if flag_test2 == True:
    T = np.linspace(start=25000, stop = 10000, num=20)
    X = (5*2*np.log(T+2)**2 + 100*2*np.log(T+2))/T + 0.1
    Label = ['$0.1+200\log(T+2)+10\log^2(T+2)$']*len(T)
    
    recorded_data = {}
            
    recorded_data['Total number of plays'] = T
        
    recorded_data['Average regret over time'] = X
    
    recorded_data['Algorithm'] = Label
    
    my_data = pd.DataFrame(recorded_data)
    
    colors = ["#4374B3"]
    # Set your custom color palette
    sns.set_palette(sns.color_palette(colors))
    
    file_name = "bound_regret"       
    plot_data_frame(my_data, xlabel="Total number of plays", 
                    ylabel="Average regret over time", huelabel='Algorithm', 
                    save_file_name=file_name, save_data_Name='test_data')

flag_test3=False
if flag_test3==True:
    plot_average_regret(start=25000, horzion=200000, nb_points=20)
    
    
    regret_data = read_data_frame('regret_data_3_alg')   
    
    T = np.exp(np.linspace(start=np.log(45000), stop = np.log(300000), num=20))
    X = (25*2*(np.log2(T+2)**2) + 100*2*np.log2(T+2)+10000)/T 
    Label = ['$O(M\log_2^{\delta}(T))$']*len(T)
    
    Dash = [1]*len(T)
    
    T = np.append(regret_data['Total number of plays'], T)
    X = np.append((regret_data['Average regret over time']), X)
    Label = np.append((regret_data['Algorithm']), Label)
    Dash = np.append([0]*len(regret_data['Algorithm']), Dash)
    
    recorded_data = {}            
    recorded_data['Total number of plays'] = T        
    recorded_data['Average regret over time'] = X    
    recorded_data['Algorithms'] = Label
    recorded_data['Dash'] = Label
    
    bound_data = pd.DataFrame(recorded_data)    
    
    g = plot_data_frame(bound_data, xlabel="Total number of plays", 
                    ylabel="Average regret over time", huelabel='Algorithms')
    
    g.ax.lines[3].set_linestyle("--")
    g.ax.lines[3].set_color("grey")
    
    g.ax.set(xscale="log")
    
    le = g.ax.legend()
    le.get_lines()[4].set_color('grey')
    le.get_lines()[4].set_linestyle("--")
    le.get_frame().set_facecolor('none')    
    le.get_frame().set_edgecolor('none')    
    
    file_path = prepare_file_name(file_name="monte_carlo_regret", ext_format='pdf', add_timestamp=False)
    g.savefig(file_path)  
    
flag_test4=True
if flag_test4==True:
    game_config = CONFIG.ENV_SCENARIO_3
    start = game_config.T_start
    nb_point = game_config.T_step
    game_horizon = game_config.game_horizon
    
    plot_repeated_simu_results(start=start, horzion=game_horizon, nbPoints=nb_point, flag_bound=True,
                                data_file_name=game_config.repeated_play_data_name)