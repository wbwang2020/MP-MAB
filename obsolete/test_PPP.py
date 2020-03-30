# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:16:15 2019

@author: wenbo2017
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt

import simu_config as CONFIG
#import matlab.engine

#from HetNetSimulator import HomeBrewedHetNetEnv
import argparse
import os
import sys

from loggingutils import info_logger

def PoissonPP( rt, Dx, Dy=None ):
    '''
    Determines the number of events `N` for a rectangular region,
    given the rate `rt` and the dimensions, `Dx`, `Dy`.
    Returns a <2xN> NumPy array.
    '''
    if Dy == None:
        Dy = Dx
        N = scipy.stats.poisson( rt*Dx*Dy ).rvs()
        x = scipy.stats.uniform.rvs(0,Dx,((N,1)))
        y = scipy.stats.uniform.rvs(0,Dy,((N,1)))
        P = np.hstack((x,y))
    return P

if __name__ == '__main__':  
#    rate, Dx = 10, 1
#    P = PoissonPP( rate, Dx ).T
#    fig, ax = plt.subplots()
#    ax = fig.add_subplot(111)
#    ax.scatter( P[0], P[1], edgecolor='b', facecolor='none', alpha=0.5 )
#    # lengths of the axes are functions of `Dx`
#    plt.xlim(0,Dx) ; plt.ylim(0,Dx)
#    # label the axes and force a 1:1 aspect ratio
##    plt.xlabel('X') ; plt.ylabel('Y') ; ax.set_aspect(1)
#    plt.title('Poisson Process {}'.format(rate))
##    savefig( 'poisson_lambda_0p2.png', fmt='png', dpi=100 )
    
    epsilon = 0.02
    nbArm = 10
    tmp_factor = 0.1
    
    current_action = 3
    
    for ii in range(10):
        prob_no_change = 1 - epsilon**(tmp_factor)
        prob_rand_action = epsilon**(tmp_factor) / (nbArm - 1)
                
        action_array = list(range(nbArm))
        prob_array = np.zeros(nbArm)
        prob_array[:] = prob_rand_action
        prob_array[current_action] = prob_no_change
                            
        action = np.random.choice(action_array, size=None, p=prob_array) 
        
        print("new action: {}; prob_stay: {:.2}, prob_rnd_change: {:.2}".format(action, prob_no_change, prob_rand_action))
    
#    test_simulator = HomeBrewHetNetEnv({'context 1'}, 10, 10)
#    test_simulator.initialize_UE(10, distance = 200, dist_mode = 0)
#    
#    test_simulator.helper_plot_ue_posiiton()
#    bs_position = [1,2]
#    bs_position = np.broadcast_to(bs_position, (10,2))
#    
#    print(bs_position)
#    eng = matlab.engine.connect_matlab()
#    eng.sqrt(4.0)
    
    C_set =  {"context 1", "context 2", "context 3"}
    
    my_logger = info_logger()      
    my_logger.logger.debug("test message.")
    
    record_series = np.empty((0,4))
    
    record1 = np.array([1, 2, 3, 4])
    record2 = np.array([0, 9, 8, 7])
    
    record_series = np.append(record_series, [record1], axis=0)
    record_series = np.append(record_series, [record2], axis=0)
    print(record_series)
    print(record_series.shape)
    
    ret_rand = np.random.uniform(low=0.5, high=1.0, size=3)
    print(ret_rand)
    
#    game_config = CONFIG.ENV_SCENARIO_3
#    print("MAB game with configuration '{}' starts to play...".format(game_config.__repr__()))
    my_parser = argparse.ArgumentParser(description='Select the configuration type to run the simulations')
    
    # Add the arguments
    my_parser.add_argument('-id',
                           metavar='ID',
                           type=int,
                           help='Choose the configuration ID between [1-6]')
    
    # Execute the parse_args() method
    args = my_parser.parse_args()

    if args.id is not None:
        print ("id has been set to {}".format(args.id))
    else:
        args.id = 1
        print ("id has been set to {}".format(args.id))