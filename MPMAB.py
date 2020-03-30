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

"""
Implementing the class 'MAB' and its children classes, which define the environment of the bandit game (stochastic i.i.d.)
"""

__author__ = "Wenbo Wang"

import numpy as np
import scipy.io
from plotutils import prepare_file_name

from Arms import UniformArm, GaussianArm

class MP_MAB(object):
    """
    i.i.d. multi-arm bandit problem. 
    The arm value is jointly sampled with the context, and for each player the underlying process may be different.
    """
    def __init__(self, context_set, nbArms, nbPlayers):
        """New MP-MAB."""
        print("\nCreating a contextual multi-player MAB game...")  # DEBUG
        
        self.nbArms = nbArms
        self.nbPlayers = nbPlayers
        
        self.context_set = context_set
        self.context_probabilites = []
        self.context_array = [] # may the context iterable
        self.flag_context_prob = False
        
        self.current_arm_value = np.zeros((nbPlayers, nbArms))
        self.current_context = None
        
        self.arms = {}
        self.max_arm_value = {} # recording the maximum arm value in case of normalization for each context along the time horizon
        
        self.horizon = 0
        self.flag_sample_prepared = False
        
    """
    For different joint distributions of (context, arm-value), we may need different initilization variables.
    Call one of the following methods for class instantiation with different types of arms instead of __init__.
    """
        
    @classmethod
    def uniform_mab(cls, context_set, nbArms, nbPlayers, dic_lower, dic_upper):
        uniform_inst = cls(context_set, nbArms, nbPlayers)
        
        # For each context and each player, we create an arm
        for context in context_set:
            player_arm_array = [[None]*nbArms for playerID in range(nbPlayers)]
            for playerID in range(nbPlayers):
                for armID in range(nbArms):
                    # if it is a uniform arm
                    param = {"lower_val": dic_lower[(context, playerID)][armID],
                             "upper_val": dic_upper[(context, playerID)][armID],
                             "context": context,
                             "playerID": playerID,
                             "armID": armID }
                    player_arm_array[playerID][armID] = UniformArm(param)
#                print("size of the object array: ", len(arm_array))#debug
            
            uniform_inst.arms[context] = player_arm_array
#            print("size of the object array for context: ", context, ": (", len(player_arm_array), ",", len(player_arm_array[0]), ")")#debug            
        
        return uniform_inst
    
    @classmethod
    def gaussian_mab(cls, context_set, nbArms, nbPlayers, dic_mean, dic_sigma):
        gaussian_inst = cls(context_set, nbArms, nbPlayers)
    
        # For each context and each player, we create an arm
        for context in context_set:
            player_arm_array = [[None]*nbArms for playerID in range(nbPlayers)]
            for playerID in range(nbPlayers):
                for armID in range(nbArms):
                    # if it is a uniform arm
                    param = {"mu": dic_mean[(context, playerID)][armID],
                             "sigma": dic_sigma[(context, playerID)][armID],
                             "context": context,
                             "playerID": playerID,
                             "armID": armID }
                    player_arm_array[playerID][armID] = GaussianArm(param)
#                print("size of the object array: ", len(arm_array))#debug
            
            gaussian_inst.arms[context] = player_arm_array
#            print("size of the object array for context: ", context, ": (", len(player_arm_array), ",", len(player_arm_array[0]), ")")#debug            
        
        return gaussian_inst

    
    def set_discrete_context_prob(self, context_prob):
        """
        assign arbitrary probabilities to contexts
        """
        if set(context_prob.keys()) != self.context_set:
            raise Exception("probability values do not match the set of context")
        
        self.context_array = np.array(list(context_prob.keys()))
        
        self.context_probabilites = np.array(list(context_prob.values()))
        self.context_probabilites = self.context_probabilites / np.sum(self.context_probabilites) # normalize
        
        self.flag_context_prob = True

    def get_discrete_context_prob(self):
        if self.flag_context_prob:
            return self.context_array, self.context_probabilites
        else:
            prob = np.ones(len(self.context_set))
            return np.array(list(self.context_set)), prob / np.sum(prob)

    """Draw samples"""
    def draw_sample(self, t=None):
         """ 
         Draw samples for all the player-arm pairs in a given sampled context.
         We enforce that the arm values are drawn in the same global context.
         """
         
         # context is finite, so here we can adopt a separate discrete (e.g., uniform) distribution for context evolution
         # in the real-world situation context-arm-value can be seen as being sampled from a joint distribution
         if self.flag_context_prob == False:
             context = np.random.choice(tuple(self.context_set)) # uniform randomly sampled
         else:
             context = np.random.choice(self.context_array, p=self.context_probabilites)
         
         player_arm_array = self.arms[context]
         for playerID in range(self.nbPlayers):
             for armID in range(self.nbArms):
                 if  player_arm_array[playerID][armID].playerID != playerID or player_arm_array[playerID][armID].armID != armID:
                     raise Exception("player ID and arm ID do not match!")
                 
                 self.current_arm_value[playerID][armID] = player_arm_array[playerID][armID].draw_sample(context, t)
         
#         print("Sampling arms completes")
         self.current_context = context

         return self.current_context,self.current_arm_value
        
    """get the samples in advance"""
    def prepare_samples(self, horizon, flag_progress_bar=False):
        if horizon <= 0:
            raise Exception("Input horizon is not valid")
                    
        self.horizon = horizon
        
        for context in self.context_set:
            for playerID in range(self.nbPlayers):
                for armID in range(self.nbArms):
                    # for each player-arm pair, prepare its sample sequences in each context
                    self.arms[context][playerID][armID].prepare_samples(horizon)
                    
            self.max_arm_value[context] = np.ones(horizon) #
                    
        self.flag_sample_prepared = True
    
    """utility functions"""
    def get_param(self, context):
         lower = np.zeros((self.nbPlayers, self.nbArms))
         upper = np.zeros((self.nbPlayers, self.nbArms))
         means = np.zeros((self.nbPlayers, self.nbArms))
         variance = np.zeros((self.nbPlayers, self.nbArms))
         
         for playerID in range(self.nbPlayers):
             for armID in range(self.nbArms):
                 lower[playerID][armID] = self.arms[context][playerID][armID].lower
                 upper[playerID][armID] = self.arms[context][playerID][armID].upper
                 means[playerID][armID] = self.arms[context][playerID][armID].mean
                 variance[playerID][armID] = self.arms[context][playerID][armID].variance
                 
         return lower, upper, means, variance
    
    def get_current_param(self, t=None):
         """ 
         Get the current sampling parameters of arms in the given context.
         """
         if self.current_context is None:
             raise Exception("The MAB game is not started.")
         
         return self.get_param(self.current_context)
        
    """
    
    """
    def save_environment(self, file_name=None):
        if self.flag_sample_prepared == False:
            print("No data is prepared")
        else:       
            # TODO: we cannot select the path yet, put the file to the default directory "\results" of the current path            
            file_path = prepare_file_name("{}-{}".format(file_name if file_name is not None else "", "env"), 
                                          alg_name = None, ext_format = "mat")
        
            mdict = {}
            for context in self.context_set:
                for playerID in range(self.nbPlayers):
                    for armID in range(self.nbArms):
                        dict_key = "{}-{}-{}".format(context, playerID, armID)
                        mdict[dict_key] = self.arms[context][playerID][armID].prepared_samples
            
            scipy.io.savemat(file_path, mdict)
        
    def load_environment(self, file_path, horizon=None):
        mdict = scipy.io.loadmat(file_path)
        
        for key in mdict:
            key_strings = key.split('_')
            context = key_strings[0]
            playerID = int(key_strings[1])
            armID = int(key_strings[2])
            
            self.arms[context][playerID][armID].prepared_samples = mdict[key]
        
        self.flag_sample_prepared = True
        
# ploting methods