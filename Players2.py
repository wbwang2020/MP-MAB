# -*- coding: utf-8 -*-
"""
@author: Wenbo Wang

[Wang2020] Wenbo Wang, Amir Leshem, Dusit Niyato and Zhu Han, "Decentralized Learning for Channel 
Allocation inIoT Networks over Unlicensed Bandwidth as aContextual Multi-player Multi-armed Bandit Game"

License:
This program is licensed under the GPLv2 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

# This file defines the player behavior for the specific SOC-MP-MAB algorithms (SOC in MABAlgorithms2.py)
# see also Players.py for other algorithms

__author__ = "Wenbo Wang"

import numpy as np
from Players import Player

if __name__ == '__main__':
    print("Warning: this script 'PlayerResult2.py' is NOT executable..")  # DEBUG
    exit(0)


class SOCPlayer(Player):

    def __init__(self, param):
        """ 
        SOCPlayer is the player for the algorithm "stable orthogonal allocation (SOC)" proposed in
        "Multi-player multi-armed bandits for stable allocation in heterogeneous ad-hoc networks", IEEE JSAC oct. 2019,
        Sumit J. Darak and Manjesh K. Hanawal [Sumit2019].
        
        The algorithm is featured by a protocol explicitly resolving collisions with channel switching,
        and the channel statistics (index) is learned based on upper confidence bound (UCB).
        
        Channel allocation is obtained through a master-slave allocation process. Social optimality is not guaranteed.
        """
        self.nbArm = param["nbArm"]
                    
        #for arm of a specific context-player
        self.playerID = param["playerID"]
    
        self.flag_master = False
        self.master_collision = np.zeros(2)
        
        self.flag_lock = False
    
        self.selected_arm = 0     
        self.policy = -1 # set to an invalid value
        
        self.time = 0
        self.accumulated_value = np.zeros(self.nbArm)
        self.arm_score = np.zeros(self.nbArm) # for UCB score computation
        self.nb_observation = np.zeros(self.nbArm) # number of observed non-zero payoff
        self.ranked_armIDs = np.array(list(range(0, self.nbArm))) #ranked according to UCB score 
        
        self.flag_agree_switching = 0 # a 3-state flag, -1: not agree, 0: irrelavent, 1: agree
        
    def reset(self):
        self.flag_master = False
        self.flag_lock = False
        self.flag_agree_switching = 0
    
        self.selected_arm = 0
        self.policy = -1 # set to an invalid value
        
        self.time = 0        
        
        self.accumulated_value = np.zeros(self.nbArm)
        self.arm_score = np.zeros(self.nbArm) # for UCB score computation
        self.nb_observation = np.zeros(self.nbArm) # number of observed non-zero payoff
        self.ranked_armIDs = np.array(list(range(0, self.nbArm))) #ranked according to UCB score 
        self.master_collision[:] = 0
        
    # --- functionalities
    def explore(self, context = None, time = None):
        """
        explore() is equivalent to the algorithm "Random Hopping" in [Sumit2019], 
        it allows users to orthogonalize on channels through uniformly drawing action samples at random  
        
        flag_lock has to be set after observing the collision feedback
        """
        if self.flag_lock == True:
            # choose the same action, do nothing
            if self.policy == -1:
                self.policy = self.selected_arm            
        else:
            self.selected_arm = np.random.randint(self.nbArm)
            
        return self.selected_arm    
            
    def learn_arm_value(self, context = None, arm_values = None, collisions = None):
        # UCB score
        if self.flag_agree_switching == -1:
            # no arm is selected, this case happens when a slave node evacuates a channel
            # to notify the master that it won't switch
            pass
        elif collisions[self.selected_arm] == 1:
            # no collisions
            self.flag_lock = True
            
            self.time = self.time + 1 # only increment when a good sample is obtained
            
            armID = self.selected_arm
            self.nb_observation[armID] = self.nb_observation[armID] + 1                
            self.accumulated_value[armID] = self.accumulated_value[armID] + arm_values[armID]
            
            # update UCB Scores
            self.arm_score = self.accumulated_value / (self.nb_observation+1e-9) + np.sqrt(2*self.time / (self.nb_observation+1e-9))
            # get the preference
            self.ranked_armIDs = np.argsort(-self.arm_score)
        
    def exploit(self, context=None, time=None):
        # SOC doesn't have a clear phase of exploitation, the players uses a collision avoidance-like
        # protocol to explicitly allocate the channels among players
        assert self.policy != -1, "policy is not obtained"
        
        self.selected_arm = self.policy        
        return self.selected_arm
            
    def set_master(self, MB_id):
        assert self.flag_lock == True, "the channel is not locked yet"
        
        # check if the MB_id is currently self.selected_arm        
        if self.policy == MB_id:
#            print("set_master(): master node ID {} at MB {}".format(self.playerID, MB_id)) # debugging
            self.flag_master = True
            # reset the recorder
            self.master_collision[:] = 0
        else:
#            print("set_master: slave node ID {} at MB {}".format( self.playerID, MB_id)) # debugging
            self.flag_master = False
            
        return self.flag_master
        
    def set_master_action(self, SB_id):
        """
        set the action of the master node (as the channel indicated by the current block ID)
        """        
        assert self.flag_lock == True, "the channel is not locked yet"
        assert  self.flag_master == True, "not a master node"
        assert self.policy == self.selected_arm, "action not aligned to policy"
        
        # get the ranked_arms without self.selected_arm
        tmp_arm_rank = np.ndarray.tolist(self.ranked_armIDs)
        
        # see footnote 1 of [Sumit2019]
        current_arm_rank = tmp_arm_rank.index(self.selected_arm)
        tmp_arm_rank.pop(current_arm_rank)       
        
        if SB_id - 1 < current_arm_rank:              
#            print("Master ID-{}: av-{:.2} ---> av-{:.2}".format(self.playerID, self.arm_score[self.selected_arm], 
#                  self.arm_score[tmp_arm_rank[SB_id - 1]])) # debugging
            master_arm_choice = tmp_arm_rank[SB_id-1]
        else:
            master_arm_choice = self.selected_arm
            
        # set policy to the currently reserved channel, signal over the new channel
        self.policy = self.selected_arm
        self.selected_arm = master_arm_choice
        
        return self.selected_arm, self.policy # new, old (MB)
        
    def decide_switching(self, subslot_id, target_arm=None):
        # has to be called by a slave
        assert self.flag_lock == True, "the channel is not locked yet"                
        assert self.flag_master == False, "not a slave node."
        assert self.policy != -1, "policy is not set"
        
        if subslot_id == 0:
            # it is in a channel transmit (CT) sub-slot
            assert target_arm is not None, "master arm choice not set"
 
            if target_arm != self.selected_arm:
                # not requested and do nothing                 
                self.flag_agree_switching = 0 # not requested
                
#                print("Slave ID-{}: not requested {} ---> {}".format(self.playerID, self.selected_arm, target_arm)) # debugging
            else:                            
                arm_rank_list = np.ndarray.tolist(self.ranked_armIDs)
                current_arm_rank = arm_rank_list.index(self.selected_arm)
                requested_arm_rank = arm_rank_list.index(target_arm)
                
#                print("Slave ID-{}: av-{:.2} ---> av-{:.2}".format(self.playerID, self.arm_score[self.selected_arm],
#                      self.arm_score[target_arm])) # debugging
                
                if requested_arm_rank < current_arm_rank:
                    # if master_arm_choice has a higher score, switch
                    self.flag_agree_switching = 1 # agreed   
                    
                    self.selected_arm = self.policy # choose the currently preferred arm
                    self.policy = target_arm # update policy
                    
#                    print("UE-{} agrees: CH-{} to CH-{} w/ scores: {} to {}".format(self.playerID, 
#                          self.selected_arm, target_arm, arm_rank_list[current_arm_rank], arm_rank_list[requested_arm_rank])) # debugging
                else:
                    # if master_arm_choice is worse than the current arm, refuse switching
                    self.flag_agree_switching = -1 # refused
                    # no change to policy  
                    self.selected_arm = self.policy
        else:
            # it is in a channel switch (CS) sub-slot
            if self.flag_agree_switching == -1:
                # refuse swithcing, leave the channel for one slot
                self.selected_arm = -1
            else:
                # if self.flag_agree_switching == 1: # agree to switch, stay on the channel to collide
                # if self.flag_agree_switching == 0: # not requested to switch, stay on the channel
                # transmit on the same channel or not affected
                pass
                    
        return self.selected_arm
    
    def update_policy(self, subslot_id, collisions):
        # the original paper does not specify when to stop updating the arm-value estimation
        # so we aussme that it never stops        
        assert self.flag_lock == True, "the channel is not locked yet"            
        
        # update actions
        if subslot_id == 0:
            # only update the master in CS slot        
            if self.flag_master == True:
                self.master_collision[0] = collisions[self.selected_arm]
            else:
                pass 
        elif subslot_id == 1:
            if self.flag_master == True:
                self.master_collision[1] = collisions[self.selected_arm]
                
                #update policy and action, according to Fig.2 [Sumit2019]
                if self.master_collision[0] > 1 and self.master_collision[1] > 1: # senario 1 (colliding twice): 
                    # switching allowed
                    self.policy = self.selected_arm
                    
#                    print("Master ID-{}: policy updated w/ switching".format(self.playerID)) # debugging
                elif self.master_collision[0] == 1 and self.master_collision[1] == 1: # senario 3 (no collision, twice)
                    self.policy = self.selected_arm
                    
#                    print("Master ID-{}: policy updated for vacant channel".format(self.playerID)) # debugging
                else:
                    # roll back
#                    print("Master ID-{}: policy rolled back".format(self.playerID)) # debugging
                    pass
                
                # reset the recorder
                self.master_collision[:] = 0
            else:   
                # reset flag to "not requested"
                self.flag_agree_switching = 0
        else:
            raise Exception("invalid sub-slot ID.")
            
        self.selected_arm = self.policy