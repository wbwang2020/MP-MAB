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

# This file defines and implements the multi-player, multi-arm bandits algorithms.
# Currently, the realized algorithms include:
# 1. SIC-MMAB: proposed in "SIC-MMAB: synchronisation involves communication in multiplayer multi-armed bandits.",
# by Boursier, Etienne, and Vianney Perchet, in Advances in Neural Information Processing Systems. 2019.
#
# Typically, one distributed algorithm is accompanied by a corresponding player class
# see also MABAlgorithms.py and MABAlgorithms2.py

import numpy as np
from MABAlgorithms import MABAlgorithm
from Players3 import SICMMABPlayer

from loggingutils import info_logger

if __name__ == '__main__':
    print("Warning: this script 'MABAlgorithms3.py' is NOT executable..")  # DEBUG
    exit(0)
    

"""
 Algorithm: SICMMB based on musical chairs
"""          
class SICMMAB(MABAlgorithm):
    def __init__(self, param):
        self.nbPlayer = param["nbPlayer"]
        self.nbArm = param["nbArm"]
        self.context_set = param["context_set"] # not really used by the algorithm
        self.horizon = param["horizon"]
        
        #each player will be attached a single agent
#        self.nbAgent = self.nbPlayer
        self.agents = []
        
        for playerID in range(self.nbPlayer):
            player_param = {"horizon": self.horizon, 
                            "nbArm": self.nbArm,
                            "playerID": playerID
                }
            
            if "T0" in param.keys():
                player_param["T0"] = param["T0"]
            
            self.agents.append(SICMMABPlayer(player_param))
        
    # --- Printing
    def __str__(self):
        return "SIC-MMAB"
    
     # --- functionalitiess        
    def reset(self, horizon=None):
        self.time = 0
        for agent in self.agents:
            agent.reset()
        
        if horizon is not None:
            self.horizon = horizon
            
    def learn_policy(self, game_env, context=None, time=None):
        (nbPlayer, nbArm) = np.shape(game_env)
#        print("number of arms: {}, number of recorded arms: {}".format(nbArm, self.nbArm))

        assert nbArm == self.nbArm, "input arm number does not match the stored environment parameters."        
        assert nbPlayer == self.nbPlayer, "input player number does not match the stored environment parameters."        
        assert nbPlayer <= nbArm, "player number should be larger than or equal to arm number."
        assert time is not None, "time is not given."
            
        pulls = np.zeros((nbPlayer, nbArm))
        action_vector = [0] * nbPlayer
        
        for agentID in range(nbPlayer):
            armID = self.agents[agentID].explore(context, time)
            pulls[agentID][armID] = 1  
            action_vector[agentID] = armID
                
        collisions = self.resolve_collision(pulls)
        sampled_rewards = self.observe_distributed_payoff(game_env, collisions) 

        for agentID in range(nbPlayer):
            assert action_vector[agentID] >= 0
            assert action_vector[agentID] < nbArm
            self.agents[agentID].update(action_vector[agentID], sampled_rewards[agentID], collisions[action_vector[agentID]])
                          
        total_rewards = np.sum(sampled_rewards)        
        return pulls, total_rewards, sampled_rewards