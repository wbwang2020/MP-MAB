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

# This file defines and implements the following multi-player, multi-arm bandits algorithms (see also Algorithms.py).
#
# 1. [Sumit2019] Sumit J. Darak and Manjesh K. Hanawal, "Multi-player multi-armed bandits for stable allocation in 
# heterogeneous ad-hoc networks", IEEE JSAC oct. 2019.

__author__ = "Wenbo Wang"

import numpy as np
from MABAlgorithms import MABAlgorithm
from Players2 import SOCPlayer

if __name__ == '__main__':
    print("Warning: this script 'MABAlgorithms2.py' is NOT executable..")  # DEBUG
    exit(0)

"""
 Algorithm: stable orthogonal allocation (SOC)
"""    
class SOC(MABAlgorithm):
    """ 
    SOC implements the algorithm "stable orthogonal allocation (SOC)" proposed in
    "Multi-player multi-armed bandits for stable allocation in heterogeneous ad-hoc networks", 
    IEEE JSAC oct. 2019, by Sumit J. Darak and Manjesh K. Hanawal [Sumit2019].
        
    The algorithm is featured by a protocol explicitly resolving collisions with channel switching,
    and the channel statistics (index) is learned based on upper confidence bound (UCB).
    
    However, it does not have a explicit function for when to stop exporation, as in the musical chairs.        
    Channel allocation is obtained through a master-slave allocation process, with explicit coordination,
    Exploration time needs to be given.
    """
    def __init__(self, param):
        self.nbPlayer = param["nbPlayer"]
        self.nbArm = param["nbArm"]
        self.context_set = param["context_set"] # not used
        
        self.delta = 0.1 if 'delta' not in param.keys() else param['delta']
#        self.nbAgent = self.nbPlayer
        
        self.time = 0
        self.Trh = np.ceil(np.log(self.delta/self.nbArm) / np.log(1-1/4/self.nbArm))
        self.TExploration = 3000 if "exploration_time" not in param.keys() else param["exploration_time"]

        self.agents = []        
        for playerID in range(self.nbPlayer):
            player_param = {"context_set": self.context_set, 
                            "nbArm": self.nbArm,
                            "playerID": playerID
                            }
            
            self.agents.append(SOCPlayer(player_param))

        self.OHS_step = 2*(self.nbArm ** 2)
        self.MB_step = 2*self.nbArm 
        self.SB_step = 2
        
        self.current_MB_id = -1 # set to an invalid ID
        self.current_master_node = -1 # there may not be a master node for the current MB

    # --- Printing
    def __str__(self):
        return "Static Orthogonal Allocation"

    # --- functionalities
    def reset(self, horizon=None):
        self.time = 0
        self.current_MB_id = -1 # set to an invalid ID
        self.current_master_node = -1
        for agent in self.agents:
            agent.reset()

    def learn_policy(self, game_env, context=None, time=None):
        # context is not used in SOC
        (nbPlayer, nbArm) = np.shape(game_env)
        assert nbPlayer == self.nbPlayer and nbArm == self.nbArm, "input does not match the stored environment parameters."
        assert nbPlayer <= nbArm, "player number should be larger than or equal to arm number."
                        
        self.time = self.time + 1
        
        pulls = np.zeros((nbPlayer, nbArm))        
        
        # there are three phases in the game
        if self.time < self.Trh:
            #random hopping / exploration
            for agentID in range(nbPlayer):
                armID = self.agents[agentID].explore(None, time)
                pulls[agentID][armID] = 1  
        
            collisions = self.resolve_collision(pulls)    
            for agentID in range(nbPlayer):
                self.agents[agentID].learn_arm_value(None, game_env[agentID,:], collisions)                
        elif self.time <= self.TExploration:
            # master-slave process 
            # 1 OHS block has K macro blcoks (K=nbArm)
            # 1 macro block has T_mb=2K time slots, namely, K sub-blocks of 2 slots each
            OHS_id = int(np.floor((self.time - self.Trh) / (self.OHS_step)))
            MB_id = int(np.floor((self.time - self.Trh - OHS_id*self.OHS_step) / self.MB_step)) # from 0 to nbArm-1
            SB_id = int(np.floor((self.time - self.Trh - OHS_id*self.OHS_step - MB_id*self.MB_step) / self.SB_step)) # from 0 to nbArm-1        
            subslot_id = int ((self.time - self.Trh) % 2) # 0 is the CT slot and 1 is the CS slot                      
                        
            if self.current_MB_id != MB_id:
                # one master block occupies 2*nbArm slots. Update master node ID as MB_id         
                self.current_MB_id = MB_id
                
                # there may be no master node at the given MB (transmitting on MB_id), 
                # so we initialize it to an invalid value for later state-check
                self.current_master_node = -1
                #prepare the master flags of each player, only when the master ID is updated         
                master_counter = 0
                for agentID in range(nbPlayer):
                    # reset the master flag of each node
                    ret_flag = self.agents[agentID].set_master(self.current_MB_id)
                    if ret_flag == True:
                        # if being a master, record its ID
                        self.current_master_node = agentID       
                        master_counter = master_counter + 1
                        
                assert master_counter<=1, "error: more than one master"
            
            if self.current_master_node == -1:
                # if there is no master node, the MB block is wasted, see Fig.2 [Sumit2019],
                # and for the entire 2*nbArm slots no one will change actions
                for agentID in range(nbPlayer):
                    arm_choice = self.agents[agentID].exploit()
                    pulls[agentID][arm_choice] = 1
                    
                collisions = self.resolve_collision(pulls)
                # update the UCB ranking
                for agentID in range(nbPlayer):
                     self.agents[agentID].learn_arm_value(None, game_env[agentID,:], collisions)                      
            else:                   
                # a master node exists
                if SB_id == 0:
                    # force transmission to align with the current policy at the first SB
                    for agentID in range(nbPlayer):
                        arm_choice = self.agents[agentID].exploit()
                        pulls[agentID][arm_choice] = 1
                    
                    collisions = self.resolve_collision(pulls)
                    # update the UCB ranking
                    for agentID in range(nbPlayer):
                         self.agents[agentID].learn_arm_value(None, game_env[agentID,:], collisions)                                
                else:                
                    # sub-slot CT or CS for SB=1,...nbArm-1, starting to switch channels 
                    if subslot_id == 0:
                        # in the channel transit (CT) sub-slot, the master node chooses channel SB_id to switch (notify),
                        # channel SB_id is the index in its preference list.
                        # all non-master nodes stay on the their own channels                          
                        master_action, master_policy = self.agents[self.current_master_node].set_master_action(SB_id)
                        pulls[self.current_master_node][master_action] = 1 
                                                
                        for agentID in range(nbPlayer):
                            if agentID != self.current_master_node:                                    
                                # directly get slave response (instead of getting it by observing collisions)  
                                # prepare the arm choice of the slave node for the next round                       
                                slave_action = self.agents[agentID].decide_switching(subslot_id, target_arm=master_policy)     
                                pulls[agentID][slave_action] = 1
                                
                    else: #subslot_id == 1:
                        assert subslot_id == 1, "sub-slot ID is invalid"
                        # in channel switch sub-slot, the master node tries to transmit on the channel to switch
                        # non-master nodes stays on their selected channels
                        for agentID in range(nbPlayer):
                            if agentID != self.current_master_node:   
                                # only the slave occupying the target channel needs to answer the request                            
                                arm_choice = self.agents[agentID].decide_switching(subslot_id)
                                
                                if arm_choice == -1:
                                    # use invalid choice to indicate no trnasmission
                                    pulls[agentID,:] = 0
                                else:                                
                                    pulls[agentID][arm_choice] = 1
                            else:
                                arm_choice = self.agents[agentID].selected_arm
                                pulls[agentID][arm_choice] = 1
                    
                    # observe collision
                    collisions = self.resolve_collision(pulls)
                        
                    # update the UCB ranking
                    for agentID in range(nbPlayer):
                        self.agents[agentID].learn_arm_value(None, game_env[agentID,:], collisions)                                 
                        # update policy after learning 
                        self.agents[agentID].update_policy(subslot_id, collisions)
        else:
            # exploitation (no mcuh is mentioned (theoretically) regarding the performance in [Sumit2019])
            for agentID in range(nbPlayer):
                arm_choice = self.agents[agentID].exploit()
                pulls[agentID][arm_choice] = 1
                    
            collisions = self.resolve_collision(pulls)
                     
        current_rewards = self.observe_distributed_payoff(game_env, collisions)                        
        total_rewards = np.sum(current_rewards)        
        return pulls, total_rewards, current_rewards
    
    # add other algorithms here
__all__ = ["SOC"]