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

# This file defines and implements the following multi-player, multi-armed bandits algorithms (see also MABAlgorithms2.py).
#
# 1. [Tibrewal2019] Tibrewal, H., Patchala, S., Hanawal, M. K., and Darak, S. J. (2019). "Multiplayer multiarmed bandits for optimal 
#    assignment in heterogeneous networks," arXiv preprint arXiv:1901.03868,
# 
# which is used as an additional reference for hopping-based algorithms such as (see also MABAlgorithms2.py)
# 1. [Sumit2019] Sumit J. Darak and Manjesh K. Hanawal, "Multi-player multi-armed bandits for stable allocation in 
#    heterogeneous ad-hoc networks", IEEE JSAC oct. 2019.

__author__ = "Wenbo Wang"

import numpy as np
from numfi import numfi
from scipy.optimize import linear_sum_assignment

from MABAlgorithms import MABAlgorithm
from Players2a import ESEPlayer

from loggingutils import info_logger

if __name__ == '__main__':
    print("Warning: this script 'MABAlgorithms2a.py' is NOT executable...")  # DEBUG
    exit(0)

"""
 Algorithm: Explore-Signal-Exploit (ESE)
"""    
class ESE(MABAlgorithm):
    """ 
    ESE implements the algorithm "Explore-Signal-Exploit (ESE)" proposed in 
    "Multiplayer multiarmed bandits for optimal assignment in heterogeneous networks," arXiv preprint arXiv:1901.03868, 
    also by the group of Sumit J. Darak and Manjesh K. Hanawal [Tibrewal2019].
        
    The algorithm is featured by a protocol of exchanging the local arm-value estimation among players, through an additional
    operation of "observation". After acquiring the estimated arm-value matrix, each player employ the Hungarian algrithm to 
    find the optimal arm to pull.
    
    Strictly speaking, ESE is not a purely decentralized algorithm since it needs sequentially sending information from one player
    to another. It may not achieve the same performance as the rest of algorithm within the same time T, due to the extra signaling phase.
    """

    def __init__(self, param):
        self.nbPlayer = param["nbPlayer"]
        self.nbArm = param["nbArm"]
        self.context_set = param["context_set"] # not used
        
        self.delta_R = 0.1 if 'delta_R' not in param.keys() else param['delta_R'] # allowable prob. of non-orthogonal allocation
        self.epsilon = 0.1
        self.time = 0
        
        # epoch parameters (well the reference paper does not provide a clear way of determining Ts and Tb, see their Alg. 2)
        # we use a simplified appraoch
        self.Tr = np.ceil(np.log(self.delta_R / self.nbArm) / (np.log(1-1/(4*self.nbArm))))  #rounds for random hopping (Tr), given by the configuration

        self.Ts = np.ceil(8.*self.nbPlayer**2/(self.epsilon**2))
        self.Tb = np.ceil(np.log2(4.*self.nbPlayer/self.epsilon))
        
        self.current_epoch = 1
        self.round_to_last_epoch = self.Tr + self.nbArm

        self.agents = []        
        for playerID in range(self.nbPlayer):
            player_param = {"context_set": self.context_set, 
                            "nbArm": self.nbArm,
                            "nbPlayer": self.nbPlayer,
                            "playerID": playerID
                            }
            
            self.agents.append(ESEPlayer(player_param))

        info_logger().log_info('ESE random hopping phase length {}'.format(self.Tr + self.nbArm)) #debug
        info_logger().log_info('ESE sequential hopping phase length Ts {}'.format(self.Ts)) #debug
        info_logger().log_info('ESE signaling length Tb {}'.format(self.Tb)) #debug
        
        epoch_length = self.nbArm * self.Ts + self.nbPlayer * self.nbArm * self.Tb + int(np.exp(self.current_epoch))
        info_logger().log_info('ESE play epoch length {}'.format(epoch_length)) #debug

    # --- Printing
    def __str__(self):
        return "Explore Signal Exploitn"

    # --- functionalities
    def reset(self, horizon=None):
        self.time = 0
        self.current_epoch = 1
        self.round_to_last_epoch = self.Tr + self.nbArm
        
        for agent in self.agents:
            agent.reset()
        
    def learn_policy(self, game_env, context=None, time=None):
        # context is not used in ESE
        (nbPlayer, nbArm) = np.shape(game_env)
        assert nbPlayer == self.nbPlayer and nbArm == self.nbArm, "input does not match the stored environment parameters."
        assert nbPlayer <= nbArm, "player number should be larger than or equal to arm number."
                        
        self.time = self.time + 1
        
        pulls = np.zeros((nbPlayer, nbArm))       
        
        # there is a single random hopping phases in the game, arm-value is not learned in this phase
        if self.time <= self.Tr:
            for agentID in range(nbPlayer):
                armID = self.agents[agentID].random_hop(None, time)
                pulls[agentID][armID] = 1  
            
            collisions = self.resolve_collision(pulls)
        
        elif self.time <= self.Tr + self.nbArm:
            # this phase is to simulate the process for players to estimate the number of players
            for agentID in range(nbPlayer):
                pulls[agentID][self.agents[agentID].current_arm] = 1 
                
            collisions = self.resolve_collision(pulls)
            
        elif self.time - self.round_to_last_epoch <= self.nbArm * self.Ts:
            #exploration with sequential hopping
            for agentID in range(nbPlayer):
                armID = self.agents[agentID].sequential_hop(None, time)
                pulls[agentID][armID] = 1
                   
            collisions = self.resolve_collision(pulls)
            for agentID in range(nbPlayer):
                self.agents[agentID].learn_arm_value(None, game_env[agentID,:], collisions)                                
             
        elif self.time - self.round_to_last_epoch - self.nbArm * self.Ts <= self.nbPlayer * self.nbArm * self.Tb:
            # signaling phase
            # we don't need to simulate the complete signaling process, but we need to truncate the prevision of each player's estimated arm-values 
            
            # the original phase of signaling, according to [Tibrewal2019], does not contributes to the accumulated sum (experience) of arm evaluation,
            # since for most of the time one player needs to either "observe" a certain arm without playing one, or stop to play to "broadcast" bit "0" to the other players.
            # However, this incurs HUGE amount of regret since the signaling phase is too long to neglect. 
            # This may be a flaw of the original design of the ESE algorithm. 
            # In our implementation we assume that the players are still able to get a reward during signaling phase. 
            
            if self.time - self.round_to_last_epoch - self.nbArm * self.Ts == 1:
                arm_matrix = np.zeros((self.nbPlayer, self.nbArm))
                for agentID in range(nbPlayer):
                    arm_matrix[agentID, :] = self.agents[agentID].arm_score
                            
                truncated_arm_matrix = numfi(arm_matrix, bits_frac=int(np.log2(4*self.nbPlayer/self.epsilon)))
                        
                for agentID in range(nbPlayer):
                    self.agents[agentID].estimated_arm_matrix = truncated_arm_matrix
                    self.agents[agentID].estimated_arm_matrix[agentID, :] = self.agents[agentID].arm_score
                            
                    # each player performs local Hungarian algorithm to derive its "optimal" policy
                    # the mehtod requires the number of rows (jobs) to be larger than that of columns (workers)
                    cost_matrix = np.negative(self.agents[agentID].estimated_arm_matrix.transpose())
                    # note that the cost_matrix is a transpose of the original matrix
                    col_ind, row_ind = linear_sum_assignment(cost_matrix) 
                            
                    # set player's policy
                    for ii in range(len(row_ind)):
                        playerID = row_ind[ii]
                        if playerID == agentID:
                            self.agents[agentID].policy = col_ind[ii]
                            pulls[agentID][col_ind[ii]] = 1
            
            for agentID in range(self.nbPlayer):
                armID = self.agents[agentID].policy
                pulls[agentID][armID] = 1  
            
            collisions = self.resolve_collision(pulls)  
            
        elif self.time - self.round_to_last_epoch - self.nbArm * self.Ts - self.nbPlayer * self.nbArm * self.Tb <= int(np.exp(self.current_epoch)):
            # exploitation phase
            for agentID in range(nbPlayer):
                armID = self.agents[agentID].exploit(context, self.time)
                pulls[agentID][armID] = 1  
                
            collisions = self.resolve_collision(pulls)              
                              
            if self.time == self.round_to_last_epoch + self.nbArm * self.Ts + self.nbPlayer * self.nbArm * self.Tb + int(np.exp(self.current_epoch)):
                #update round number
                self.round_to_last_epoch += self.nbArm * self.Ts + self.nbPlayer * self.nbArm * self.Tb + int(np.exp(self.current_epoch))
                self.current_epoch = self.current_epoch + 1
                    
                info_logger().log_info('ESE play epoch {}'.format(self.current_epoch)) #debug
            
        current_rewards = self.observe_distributed_payoff(game_env, collisions)                        
        total_rewards = np.sum(current_rewards)        
        return pulls, total_rewards, current_rewards
    
# add other algorithms here
__all__ = ["ESE"]        