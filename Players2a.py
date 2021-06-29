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

from loggingutils import info_logger

if __name__ == '__main__':
    print("Warning: this script 'PlayerResult2.py' is NOT executable..")  # DEBUG
    exit(0)
    
    
class ESEPlayer(Player):
    """ 
    ESEPlayer is the player for the algorithm "Explore-Signal-Exploit (ESE)" proposed in
    "Multiplayer multiarmed bandits for optimal assignment in heterogeneous networks," arXiv preprint arXiv:1901.03868, 
    by Sumit J. Darak and Manjesh K. Hanawal [Tibrewal2019].
        
    The algorithm is featured by a protocol using player's state to carry the data load for signaling
    """
    def __init__(self, param):
        self.nbArm = param["nbArm"]
        self.nbPlayer = param["nbPlayer"]
        
#        info_logger().log_info('ESE player number {}'.format(self.nbPlayer)) #debug
                    
        #for arm of a specific context-player
        self.playerID = param["playerID"]
        
        self.flag_lock = False
    
        self.selected_arm = 0 # index of the locked arm
        self.policy = -1 # set to an invalid value
        self.current_arm = 0
        self.arm_score = np.zeros(self.nbArm)
        self.estimated_arm_matrix = np.zeros((self.nbPlayer, self.nbArm))
        
        self.accumulated_value = np.zeros(self.nbArm)
        self.nb_observation = np.zeros(self.nbArm) # number of observed non-zero payoff
        
    def reset(self):
        self.flag_lock = False
    
        self.selected_arm = 0
        self.policy = -1 # set to an invalid value
        self.current_arm = 0
        
        self.arm_score = np.zeros(self.nbArm) 
        self.estimated_arm_matrix = np.zeros((self.nbPlayer, self.nbArm))
        
        self.accumulated_value = np.zeros(self.nbArm)
        self.nb_observation = np.zeros(self.nbArm) # number of observed non-zero payoff

        
     # --- functionalities
    def random_hop(self, context = None, time = None):
        """
        random_hop() is equivalent to the algorithm "Random Hopping" in [Tibrewal2019], 
        it allows users to orthogonalize on channels through uniformly drawing action samples at random  
        
        flag_lock has to be set after observing the collision feedback
        """
        if self.flag_lock == True:
            # choose the same action, do nothing
            if self.policy == -1:
                self.policy = self.selected_arm            
        else:
            self.selected_arm = np.random.randint(self.nbArm)
            
        self.current_arm = self.selected_arm 
            
        return self.selected_arm    
    
    def sequential_hop(self, context = None, time = None):
        self.current_arm = (self.current_arm + 1) % self.nbArm
        
        return self.current_arm
    
    def learn_arm_value(self, context = None, arm_values = None, collisions = None):
        # estimated arm score
        armID = self.current_arm
            
        self.nb_observation[armID] = self.nb_observation[armID] + 1                
        self.accumulated_value[armID] = self.accumulated_value[armID] + arm_values[armID]
            
        # update UCB Scores
        self.arm_score = self.accumulated_value / (self.nb_observation+1e-9)
        
    def exploit(self, context = None, time=None):       
#        self.selected_arm = self.get_best_policy(context) # if turning this on, we'll compute the best policy each time
        
        self.current_arm = self.policy
        return self.current_arm #return the action