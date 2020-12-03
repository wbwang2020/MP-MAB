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

# This file defines the player behavior for the specific SIC-MMAB algorithms (SICMMAB in MABAlgorithms3.py)
# see also Players.py, Players2.py for other algorithms
# THe algorithm implementation is based on library sic-mmab @https://github.com/eboursier/sic-mmab

__author__ = "Wenbo Wang"

import numpy as np
from Players import Player

if __name__ == '__main__':
    print("Warning: this script 'Player3.py' is NOT executable..")  # DEBUG
    exit(0)
    
STATE_FIXATION = 0
STATE_ESTIMATION = 1
STATE_EXPLOITATION = 2
STATE_COMMUNICATION = 3
STATE_EXPLORATION = 4

class SICMMABPlayer(Player):

    def __init__(self, param):
        self.horizon = param["horizon"]  #: if the horizon is not known in advance, set it to None.
        self.nbArm_fixed = param["nbArm"] # real arm number
        self.nbArm = self.nbArm_fixed # acitve arms
                    
        #for arm of a specific context-player
        self.context = None # not used by the player
        self.playerID = param["playerID"]
        
        self.time = 0   
        self.T0 = np.ceil(self.nbArm * np.e * np.log(self.horizon)) # length of T0 in Musical Chairs in initialization
        
        self.ext_rank = -1 # -1 until known
        self.int_rank = 0 # starts index with 0 here
        self.nb_active_player = 1 # number of active players
        self.last_action = np.random.randint(self.nbArm) # last play for sequential hopping
        self.selected_arm = 0
        
        self.state = STATE_FIXATION
        self.t_state = 0 # step in the current phase
        self.round_number = 0 # phase number of exploration phase 
        self.active_arms = np.arange(0, self.nbArm)
        self.sums = np.zeros(self.nbArm) # means*np_pulls
        self.last_phase_stats = np.zeros(self.nbArm)
 
        self.nb_pulls = np.zeros(self.nbArm) # number of pulls for each arm

    def reset(self):
        self.nbArm = self.nbArm_fixed
        
        self.time = 0
        self.T0 = np.ceil(self.nbArm * np.e * np.log(self.horizon)) # length of T0 in Musical Chairs in initialization
        
        self.ext_rank = -1 # -1 until known
        self.int_rank = 0 # starts index with 0 here
        self.nb_active_player = 1 # number of active players
        self.last_action = np.random.randint(self.nbArm) # last play for sequential hopping
        self.selected_arm = 0
        
        self.state = STATE_FIXATION
        self.t_state = 0 # step in the current phase
        self.round_number = 0 # phase number of exploration phase 
        self.active_arms = np.arange(0, self.nbArm)
        self.sums = np.zeros(self.nbArm) # means*np_pulls
        self.last_phase_stats = np.zeros(self.nbArm)
        
        self.nb_pulls = np.zeros(self.nbArm) # number of pulls for each arm
        
    def explore(self, context = None, time = None):
        """
        return arm to pull based on past information (given in function update)
        """
        #including the phases of initialization, estimation, exploration and communication
        if self.state == STATE_FIXATION:
            if self.ext_rank==-1: # still trying to fix to an arm
                self.selected_arm = np.random.randint(self.nbArm)
                return self.selected_arm
            else: # fix
                self.selected_arm = self.ext_rank
                return self.selected_arm
            
        # estimation of internal rank and number of players
        if self.state == STATE_ESTIMATION:
            if self.time <= self.T0 + 2*self.ext_rank: # waiting its turn to sequential hop
                self.selected_arm = self.ext_rank
                return self.selected_arm
            else: # sequential hopping
                self.selected_arm = (self.last_action+1)%self.nbArm
                return self.selected_arm
            
        # exploration phase
        if self.state == STATE_EXPLORATION:
            last_index = np.where(self.active_arms == self.last_action)[0][0]
            
            self.selected_arm = self.active_arms[(last_index+1)%self.nbArm] # sequentially hop
            return self.selected_arm
        
        # communication phase        
        if self.state == STATE_COMMUNICATION:
            if (self.t_state < (self.int_rank+1)*(self.nb_active_player-1)*self.nbArm*(self.round_number+2) and 
                (self.t_state >= (self.int_rank)*(self.nb_active_player-1)*self.nbArm*(self.round_number+2))): 
                # your turn to communicate
                # determine the number of the bit to send, the channel and the player
                
                # the actual time step in the communication phase (while giving info)
                t0 = self.t_state % ((self.nb_active_player-1)*self.nbArm*(self.round_number+2)) 
                
                b = (int)(t0 % (self.round_number+2)) # the number of the bit to send
                
                k0 = (int)(((t0-b)/(self.round_number+2))%self.nbArm) # the arm to send
                k = self.active_arms[k0]
                if (((int)(self.last_phase_stats[k])>>b)%2): # has to send bit 1
                    j = (t0-b-(self.round_number+2)*k0)/((self.round_number+2) * self.nbArm) # the player to send
                    j = (int)(j + (j>= self.int_rank))
                    
                    self.selected_arm = self.active_arms[j]
                    return self.selected_arm # send 1
                else:
                    self.selected_arm = self.active_arms[self.int_rank]
                    return self.selected_arm # send 0
                
            else:
                self.selected_arm = self.active_arms[self.int_rank]
                return self.selected_arm # receive protocol or wait
            
        # exploitation phase
        if self.state == STATE_EXPLOITATION:
            self.selected_arm = self.last_action
            return self.selected_arm
            
    def update(self, action, reward, collision):
        self.last_action = action
        
        if self.state == STATE_FIXATION:
            if self.ext_rank==-1:
                if collision == 1: # succesfully fixed during Musical Chairs
                    self.ext_rank = action
                    
            # end of Musical Chairs
            if self.time == self.T0:
                self.state = STATE_ESTIMATION # estimation of M
                self.last_action = self.ext_rank
                               
        elif self.state == STATE_ESTIMATION:
            if collision > 1: # collision with a player
                if self.time <= self.T0 + 2*self.ext_rank: # increases the internal rank
                    self.int_rank += 1
                self.nb_active_player += 1 # increases number of active players
                
            # end of initialization
            if self.time == self.T0 + 2*self.nbArm:
                self.state = STATE_EXPLORATION
                self.t_state = 0
                
                # we actually not start at the phase p=1 to speed up the exploration, without changing the asymptotic regret
                self.round_number = (int)(np.ceil(np.log2(self.nb_active_player))) 
                    
        elif self.state == STATE_EXPLORATION:
            self.last_phase_stats[action] += reward # update stats
            self.sums[action] += reward
            self.t_state += 1
            
            # end of exploration phase
            if self.t_state == (2<<self.round_number) * self.nbArm: 
                self.state = STATE_COMMUNICATION
                self.t_state = 0         
            
        elif self.state == STATE_COMMUNICATION:
                # reception case
            if (self.t_state >= (self.int_rank+1)*(self.nb_active_player-1)*self.nbArm*(self.round_number+2) or 
                (self.t_state < (self.int_rank)*(self.nb_active_player-1)*self.nbArm*(self.round_number+2))):
                if collision > 1: # collision with a player
                    # the actual time step in the communication phase (while giving info)
                    t0 = self.t_state % ((self.nb_active_player-1)*self.nbArm*(self.round_number+2)) 
                    
                    b = (int)(t0 % (self.round_number+2)) # the number of the bit to send

                    k0 = (int)(((t0-b)/(self.round_number+2))%self.nbArm) # the channel to send
                    k = self.active_arms[k0]
                
                    self.sums[k] += ((2<<b)>>1)
                          
            self.t_state += 1
            
            # end of the communication phase
            # update many things
            if (self.t_state == (self.nb_active_player)*(self.nb_active_player-1)*self.nbArm*(self.round_number+2) 
                or self.nb_active_player==1):

                # update centralized number of pulls
                for k in self.active_arms:
                    self.nb_pulls[k] += (2<<self.round_number)*self.nb_active_player
                
                # update confidence intervals
                b_up = (self.sums[self.active_arms]/self.nb_pulls[self.active_arms] + 
                        np.sqrt(2*np.log(self.horizon)/(self.nb_pulls[self.active_arms])))
                b_low = (self.sums[self.active_arms]/self.nb_pulls[self.active_arms] - 
                         np.sqrt(2*np.log(self.horizon)/(self.nb_pulls[self.active_arms])))
                reject = []
                accept = []

                # compute the arms to accept/reject    
                for i, k in enumerate(self.active_arms):
                    better = np.sum(b_low > (b_up[i]))
                    worse = np.sum(b_up < b_low[i])
                    if better >= self.nb_active_player:
                        reject.append(k)
                    if worse >= (self.nbArm - self.nb_active_player):
                        accept.append(k)

                # update set of active arms            
                for k in reject:
                    self.active_arms = np.setdiff1d(self.active_arms, k)
                for k in accept:
                    self.active_arms = np.setdiff1d(self.active_arms, k)

                # update number of active players and arms
                self.nb_active_player -= len(accept)
                self.nbArm -= (len(accept)+len(reject))
                    
                if len(accept)>self.int_rank: # start exploitation
                    self.state = STATE_EXPLOITATION
                    self.last_action = accept[self.int_rank]
                else: 
                    # new exploration phase and update internal rank 
                    # (old version of the algorithm where the internal rank was changed, but it does not change the results)
                    self.state = STATE_EXPLORATION
                    self.int_rank -= len(accept)
                    self.last_action = self.active_arms[self.int_rank] # start new phase in an orthogonal setting
                    self.round_number += 1
                    self.last_phase_stats = np.zeros(self.nbArm_fixed)
                    self.t_state = 0
                        
        self.time += 1
     
    def learn_arm_value(self, context = None, arm_values = None, collisions = None):
        print("learn_arm_value() is not implemented for this particular algorithm.")
        
    def exploit(self, context = None, time=None):
        print("exploit() is not implemented for this particular algorithm.")