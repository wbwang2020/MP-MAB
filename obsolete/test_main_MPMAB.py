# -*- coding: utf-8 -*-
"""
@author: Wenbo Wang

License:
This program is licensed under the GPLv2 license. If you in any way use this code for research 
that results in publications, please cite our original article listed above.
 
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
See the GNU General Public License for more details.


This file tests the running framework of the bandit simulation
"""

__author__ = "Wenbo Wang"

import numpy as np

from MPMAB import MP_MAB
from HetNetSimulator import HomeBrewedHetNetEnv
from PlayResult import ResultMultiPlayers
from MABAlgorithms import Hungarian, MusicalChairs, TrialandError, GameofThrone
from Arms import *

import time
from tqdm import tqdm

if __name__ == '__main__':
    # test code
    horizon = 10000# should not be less than 100000 for MC    

    context_set = {"context 1", "context 2", "context 3"}
    
#    nb_player = 2
#    nb_arms = 3
#    dic_lower = {("context 1", 0): np.array([0., 0.5, 0.3]), ("context 2", 0): np.array([0.1, 0.2, 0.2]), ("context 3", 0): np.array([0., 0.2, 0.25]),
#                     ("context 1", 1): np.array([0.1, 0.6, 0.2]), ("context 2", 1): np.array([0., 0., 0.]), ("context 3", 1): np.array([0.2, 0.1, 0.45])}
#    dic_upper =  {("context 1", 0): np.array([0.5, 0.8, 0.6]), ("context 2", 0): np.array([1., 1., 0.4]), ("context 3", 0): np.array([1, 0.3, 0.65]),
#                     ("context 1", 1): np.array([0.81, 0.96, 0.52]), ("context 2", 1): np.array([0.5, 0.4, 0.9]), ("context 3", 1): np.array([0.62, 0.21, 0.95])}

    nb_player = 5
    nb_arms = 6
    
    """ 
    (1) Create an environment instance (e.g., with uniform arms) of the MPMAB    
    """
    hetnet_params = {'enabel mmWave': True,
                             'horizon': horizon,
                             'cell range': 200,
                             'context_prob': {'context 1':2, 'context 2':1, 'context 3':1},
                             'los_prob':  {'context 1':2, 'context 2':1, 'context 3':1}
            }
    multi_player_MAB = HomeBrewedHetNetEnv.HetNet_mab(context_set, nb_arms, nb_player, hetnet_params)
#    multi_player_MAB = MP_MAB.gaussian_mab(context_set, nb_arms, nb_player, dic_lower, dic_upper)
    
    multi_player_MAB.prepare_samples(horizon)
    multi_player_MAB.save_environment()

    start_time = time.time()
        
    """
    (2) Create Musical Chairs algorithm
    """
    alg_param_mc = {"nbPlayer": nb_player,
                 "nbArm": nb_arms,
                 "context_set": context_set,
                 "horizon": horizon,
                 "T0": 3000
                 }
    alg_MC = MusicalChairs(alg_param_mc)
            
    # to record the learning results of alg_MC
    result_MC = ResultMultiPlayers("Musical Chair", context_set, nb_player, nb_arms, horizon)
    
    """
    (3) Create Hungarian algorithm
    """
    alg_param_hungarian = {"nbPlayer": nb_player,
                 "nbArm": nb_arms,
                 "context_set": context_set
            }
    
    alg_hungarian = Hungarian(alg_param_hungarian)
    
#    dic_pulls_on_means = dict()
#    dic_total_rewards_on_means = dict()
#    dic_sampled_rewards_on_means = dict()
#    #get static allocation w.r.t. the means in each context
#    for context in context_set:
#        lower, upper, means, variance = multi_player_MAB.get_param(context)
#        static_pulls, static_total_reward, static_sampled_rewards = alg_hungarian.learn_policy(means)
#        
#        dic_pulls_on_means[context] = static_pulls
#        dic_total_rewards_on_means[context] = static_total_reward
#        dic_sampled_rewards_on_means[context] = static_sampled_rewards
#    
#    #recorder of learning results
#    # to store the centralized algorithm result of alg_hungarian
    result_hungarian = ResultMultiPlayers("Instant Hungarian", context_set, nb_player, nb_arms, horizon) 
#    result_hungarian_mean = ResultMultiPlayers("Hungarian", context_set, nb_player, nb_arms, horizon)
    
    """
    (4) Create trial-and-error algorithm
    """
    alg_param_tne = {"nbPlayer": nb_player,
                 "nbArm": nb_arms,
                 "context_set": context_set,
                 "horizon": horizon,
                 "c1": 100, "c2": 200, "c3": 100, 
                 "epsilon": 0.01, "delta": 2, "xi": 0.001, 
                 "alpha11": -0.12, "alpha12": 0.15, "alpha21": -0.35, "alpha22": 0.4
            }
    alg_TnE = TrialandError(alg_param_tne)
    # to store the centralized algorithm result of alg_hungarian
    result_TnE = ResultMultiPlayers("Trial-n-Error", context_set, nb_player, nb_arms, horizon) 
    
    """
    (5) Create game-of-throne algorithm
    """
    alg_param_got = {"nbPlayer": nb_player,
                 "nbArm": nb_arms,
                 "context_set": context_set,
                 "horizon": horizon,
                 "c1": 100, "c2": 200, "c3": 100, 
                 "epsilon": 0.01, "delta": 2, "xi": 0.001,             
            }
           
    alg_GoT = GameofThrone(alg_param_got)
                        
    result_GoT = ResultMultiPlayers("Game of Throne", context_set, nb_player, nb_arms, horizon) 
    
    # Main loop of learning
    for t in tqdm(range(horizon)):
        context, arm_values = multi_player_MAB.draw_sample(t)
    
        # Hungarian algoirthm over the instantaneous samples and results
        pulls, total_reward, sampled_rewards = alg_hungarian.learn_policy(arm_values)        
        choices = alg_hungarian.pulls2choices(pulls)
        result_hungarian.store(t, context, choices, sampled_rewards, total_reward, pulls)
        
        # Hungarian algoirthm over the mean samples and results
#        static_pulls = dic_pulls_on_means[context]
#        static_choices = alg_hungarian.pulls2choices(static_pulls)
#        static_reward = dic_sampled_rewards_on_means[context]
#        static_total_reward = dic_total_rewards_on_means[context]
#        result_hungarian_mean.store(t, context, static_choices, static_reward, static_total_reward, static_pulls)
        
        # Musical-chair algorithm over the  instantaneous samples and the learning results
        pulls, total_reward, sampled_rewards = alg_MC.learn_policy(arm_values, context, t)
        choices = alg_MC.pulls2choices(pulls)
        collisions = alg_MC.resolve_collision(pulls)
        result_MC.store(t, context, choices, sampled_rewards, total_reward, pulls, collisions)
        
        # Trial-and-error algorithm over the  instantaneous samples and the learning results
        pulls, total_reward, sampled_rewards = alg_TnE.learn_policy(arm_values, context, t)
        choices = alg_TnE.pulls2choices(pulls)
        collisions = alg_TnE.resolve_collision(pulls)
        result_TnE.store(t, context, choices, sampled_rewards, total_reward, pulls, collisions)
        
        # Game of Throne 
        pulls, total_reward, sampled_rewards = alg_GoT.learn_policy(arm_values, context, t)
        choices = alg_GoT.pulls2choices(pulls)
        collisions = alg_GoT.resolve_collision(pulls)
        result_GoT.store(t, context, choices, sampled_rewards, total_reward, pulls, collisions)

    #end of play
    running_time = time.time() - start_time
    print("Simulation completes in {}s for {} rounds".format(running_time, horizon))
    
    # for debugging
    print("Trial-and-error Algorithm: {} exploration rounds, {} learning rounds, {} exploitation rounds".format(alg_TnE.nbExploration, 
          alg_TnE.nbTnE, alg_TnE.nbExploitation))
        
    print("Context 1: {}, Context 2: {}, Context 3: {}".format(result_MC.context_history.count("context 1"), 
          result_MC.context_history.count("context 2"), 
          result_MC.context_history.count("context 3")) )
    
#    result_hungarian.plot_cumu_rewards(other_results=[result_MC, result_TnE], save_fig=True, save_data=False)
    result_hungarian.plot_avg_reward(other_results=[result_MC, result_GoT, result_TnE], save_fig=True, save_data=False)
        
        
        
        
        
        
        
        
        