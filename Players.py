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

# This file defines the player behavior for a series of MP-MAB algorithms

__author__ = "Wenbo Wang"

import numpy as np

from loggingutils import info_logger

if __name__ == '__main__':
    print("Warning: this script 'Player.py' is NOT executable..")  # DEBUG
    exit(0)

class Player(object):
    """ Base class for a player class."""

    def __init__(self, param):
        """ 
        Base class for a player class.
        For clarity, we require each child class to re-implement completely the __init__() method.
        """
        self.horizon = param["horizon"]  #: if the horizon is not known in advance, set it to None.
        self.nbArm = param["nbArm"]
                
        #for arm of a specific context-player
        self.context = param["context"]
        self.playerID = param["playerID"]
        
        self.arm_estimate = np.zeros(self.nbArm)
    
    # --- Printing
    def __str__(self):
        return self.__class__.__name__    
        
    # --- functionalities
    def explore(self, context = None, time = None):
        print("decision() should be implemented for agent adopting a particular algorithm.")
        
    
    def learn_arm_value(self, context = None, arm_values = None, collisions = None):
        print("learn_arm_value() should be implemented for agent adopting a particular algorithm.")
        
    def exploit(self, context = None, time=None):
        print("exploit() should be implemented for agent adopting a particular algorithm.")
        
    def reset(self):
        print("reset() should be implemented for agent adopting a particular algorithm.")
   
class MusicChairPlayer(Player):
    """ 
    Class MusicChairPlayer for a player (agent) adopting the Music Chair algorithm.
    Implemented based on the paper "Multi-Player Bandits – a Musical Chairs Approach", by Jonathan Rosenski and Ohad Shamir @2015 [Rosenski2015] 
    (https://arxiv.org/abs/1512.02866).
    Note that this algorithm is designed for multi-player only and for contextual bandit it adapts to the condition of unobservable context.
    """    
    
    def __init__(self, param):
        self.horizon = param["horizon"]  #: if the horizon is not known in advance, set it to None.
        self.nbArm = param["nbArm"]
                    
        #for arm of a specific context-player
        self.context = None # not used by the player
        self.playerID = param["playerID"]
        
        if "epsilon" in param:
            self.epsilon = param["epsilon"]
        else:
            self.epsilon = 0.1
            
        if "delta" in param:
            self.delta = param["delta"]
        else:
            self.delta = 0.05
            
        self.accumulated_value = np.zeros(self.nbArm)
        self.arm_estimate = np.zeros(self.nbArm) # \tilde{\mu}_i in [Rosenski2015]
        self.nb_collision = 0 # number of observed collision, C_{T_0} in [Rosenski2015]
        self.nb_observation = np.zeros(self.nbArm) # number of observed non-zero payoff, o_i in [Rosenski2015]
        
        if "T0" in param.keys() and param["T0"] > 0:
            self.T0 = param["T0"]    
        else:
            self.T0 = self.get_optimalT0(self.nbArm, self.horizon, self.epsilon, self.delta)
        
        self.time = 0
        
        self.sorted_chair = None        
        self.selected_arm = 0
        
        self.flag_seated = False
        self.selected_chair = 0
        self.estimated_nbPlayer = 0
    
    def reset(self):
        self.accumulated_value = np.zeros(self.nbArm)
        self.arm_estimate = np.zeros(self.nbArm) # \tilde{\mu}_i in [Rosenski2015]
        self.nb_collision = 0 # number of observed collision, C_{T_0} in [Rosenski2015]
        self.nb_observation = np.zeros(self.nbArm) # number of observed non-zero payoff, o_i in [Rosenski2015]
        
        self.time = 0
        
        self.sorted_chair = None        
        self.selected_arm = 0
        
        self.flag_seated = False
        self.selected_chair = 0
        self.estimated_nbPlayer = 0
        
        
    def get_optimalT0(self, nbArms, horizon=None, epsilon=0.1, delta=0.05):
        """ 
        Estimate T0 for an error probability delta and a bound of gap between the rewards of N-th best arm and the (N+1)-th best arm. 
        The method is based on Theorem 1 of [Rosenski2015], which requires knowing the number of arms in the game. 
        
        Equation:
            \begin{equation}
                T_0 = \ceil{\max (\frac{K}{2})\ln(\frac{2K^2}{\delta}), \frac{16K}{\epsilon^2}\ln(\frac{4K^2}{\delta}, \frac{K^2\log(\frac{2}{\delta})}{0.02})   }
            \end{equation}
        
        Remark: note that the last term \frac{K^2\log(\frac{2}{\delta})}{0.02} was written in [Rosenski2015] as \frac{K^2\log(\frac{2}{\delta_2})}{0.02}, which is a typo.
        $\delta_2$ should be $\delta$, since $\frac{K^2\log(\frac{2}{\delta_2})}{0.02}$ is derived from $t\ge \frac{\log(2/delta)}{2\epsilon_1^2}$, where 
        $\epsilon_1^2\ge \frac{0.01}{K^2}$.
        
        Examples:
    
        - For K arms, in order to have a constant regret with error probability delta, with the gap condition epsilon, we have     
        (1) optimalT0(2, None, 0.1, 0.05) = 18459
        (2) optimalT0(6, None, 0.01, 0.05) = 76469
        (3) optimalT0(17, None, 0.01, 0.05) = 273317
        """        
    
        T0_1 = (nbArms / 2.) * np.log(2 * nbArms**2 / delta)
        T0_2 = ((16 * nbArms) / (epsilon**2)) * np.log(4 * nbArms**2 / delta)
        T0_3 = (nbArms**2 * np.log(2 / delta)) / 0.02   # delta**2 or delta_2 ? Typing mistake in their paper
        T0 = max(T0_1, T0_2, T0_3)
        
        if horizon is None:
            raise Exception("the total number of rounds is not known.")
        elif T0>= horizon:
            raise Exception("the total number of rounds is too small for exploration.")
    
        return int(np.ceil(T0))
    
    def explore(self, context = None, time = None):
        if time is None or time != self.time:
            raise Exception("Playing round does not match.")
            
        #update time
        self.time = time + 1
        
        if self.time <= self.T0:
            #pahse of exploration
            self.selected_arm = np.random.randint(self.nbArm)
            
        return self.selected_arm
    
    def learn_arm_value(self, context = None, arm_values = None, collisions = None):
        # context is not used in this algorithm
        # must be called after explore
        if len(arm_values) != self.nbArm or len(collisions) != self.nbArm:
            raise Exception("inputs are invalid.")   
        
        if self.time <= self.T0:
            # get the reward of exploration phase
            if collisions[self.selected_arm] > 1:
                #selects an arm with collision
                self.nb_collision = self.nb_collision + 1
            else:
                armID = self.selected_arm
                self.nb_observation[armID] = self.nb_observation[armID] + 1                
                self.accumulated_value[armID] = self.accumulated_value[armID] + arm_values[armID]      
    
    def exploit(self, context = None, time=None):
        if time is None or time != self.time:
            raise Exception("Playing round does not match.")
            
        #update time
        self.time = time + 1
        
        if self.time > self.T0 and self.time <=self.horizon:
            if self.sorted_chair is None:
                # prepare only once
                for armID in range(self.nbArm):
                    if self.nb_observation[armID] != 0:
                        self.arm_estimate[armID] = self.accumulated_value[armID] / self.nb_observation[armID]       
                        
                # if the estimated player nubmer is not obtained, calculate it first
                # Equation for N^* is given in Alg. 1 of [Rosenski2015]
                self.estimated_nbPlayer = int(round(1 + np.log((self.T0 - self.nb_collision) / self.T0) / np.log(1. - 1. / self.nbArm)))
                if self.estimated_nbPlayer > self.nbArm:
                    self.estimated_nbPlayer = self.nbArm # force the number of players to be less than the number of arms
                
                # sort their index by empirical arm values (means) in decreasing order
                sorted_arms = np.argsort(-self.arm_estimate)  # FIXED among the best M arms!
                self.sorted_chair = sorted_arms[:self.estimated_nbPlayer]     
                
        if self.estimated_nbPlayer == 0:
            raise Exception("estimated arm number is invalid.") 
                
        if self.flag_seated == False:            
                self.selected_chair = np.random.randint(self.estimated_nbPlayer)
                self.selected_arm = self.sorted_chair[self.selected_chair]
        else:
            pass
        
        return self.selected_arm
    
    def update_musical_chair(self, time = None, collisions = None):
        if time is None or time <= self.T0:
            raise Exception("Playing round does not match.")
            
        if self.flag_seated == False and collisions[self.selected_arm] == 1:
            self.flag_seated = True             


STATE_EXPLORE = 0
STATE_LEARN = 1
STATE_EXPLOIT = 2

STATE_CONTENT = 0
STATE_HOPEFUL = 1
STATE_WATCHFUL = 2
STATE_DISCONTENT = 3

class TnEPlayer(Player):
    """ 
    Class TnEPlayer for a player (agent) adopting the trial-and-error algorithm.
    Implemented for the paper "Distributed Learning for Interference Avoidance as aContextual Multi-player Multi-armed Bandit Game", 
    by Wenbo Wang et al. [Wang2019]
    """    
    def __init__(self, param):
        if "context_set" not in param.keys():
            raise Exception("context set is not given")
        else:
            self.context_set = param["context_set"] # has to be larger than or equal to 1
        
        self.horizon = param["horizon"] if "horizon" in param.keys() else 0
                    
        #for arm of a specific context-player
        self.playerID = param["playerID"]
        self.nbArm = param["nbArm"]
        
        #used in Eq.(6) in [Wang2019]
        self.xi = param["xi"] 
        #used in Eq. (10) and Eq. (11) in [Wang2019]
        self.epsilon = param["epsilon"] 
        
        self.rho = param["rho"] #no longer used in the new algorithm
        
        #log-linear function parameters, adopted from Young's paper "learning efficient Nash equilibrium in distributed systems"
        self.alpha11 = -0.001 if param['alpha11'] is None else param['alpha11']# F(u)<1/2M
        self.alpha12 = 0.1 if param['alpha12'] is None else param['alpha12']
        
        self.alpha21 = -0.01 if param['alpha21'] is None else param['alpha21']# G(u)<1/2
        self.alpha22 = 0.5 if param['alpha22'] is None else param['alpha22']
        
        # Initialization
        self.nb_observation = {}
        self.accumulated_value = {}
        self.arm_estimate = {}
        
        self.learning_state = {}
#        self.visit_frequency = {}
        self.ptbd_arm_value = {}
        self.selected_arm = 0
        
        self.nb_state_visit = {}
        self.nb_state_aligned = {}
        
        self.current_state = {}
        self.reference_reward = {}
        
        self.best_policy = {}
        
        for context in self.context_set:
            # for arm-value estimation
            self.nb_observation[context] = np.zeros(self.nbArm)
            self.accumulated_value[context] = np.zeros(self.nbArm)
            # the static game is formulated on arm_estimate
            self.arm_estimate[context] = np.zeros(self.nbArm)
            
            self.learning_state[context] = STATE_EXPLORE

            self.ptbd_arm_value[context] = np.zeros(self.nbArm) # perturbed arm values
        
            self.nb_state_visit[context] = np.zeros((4, self.nbArm)) # for debugging purpose
            self.nb_state_aligned[context] = np.zeros(self.nbArm)                        
            """
            One example of the intermediate states:
            --- for a game of 2 arms, we have that for a given context (payoff is stored in self.reference_reward)
            (0, 0, 0): Content, arm 0, payoff = 0,
            (1, 0, 0): Hopeful, arm 0, payoff = 0,
            (2, 0, 0): Watchful, arm 0, payoff = 0,
            (3, 0, 0): Discontent, arm 0, payoff = 0,
            
            (0, 0, 1): Content, arm 0, payoff = arm-value,
            (1, 0, 1): Hopeful, arm 0, payoff = arm-value,
            (2, 0, 1): Watchful, arm 0, payoff = arm-value,
            (3, 0, 1): Discontent, arm 0, payoff = arm-value,
            
            (0, 1, 0): Content, arm 1, payoff = 0,
            (1, 1, 0): Hopeful, arm 1, payoff = 0,
            (2, 1, 0): Watchful, arm 1, payoff = 0,
            (3, 1, 0): Discontent, arm 1, payoff = 0,
            
            (0, 1, 1): Content, arm 1, payoff = arm-value,
            (1, 1, 1): Hopeful, arm 1, payoff = arm-value,
            (2, 1, 1): Watchful, arm 1, payoff = arm-value,
            (3, 1, 1): Discontent, arm 1, payoff = arm-value,
            
            """
            
            self.current_state[context] = [STATE_DISCONTENT, 0] #set as a default 3-tuple: (mood, reference action, reference payoff = 0)
            self.reference_reward[context] = 0# record the real reference reward of the state
            
            self.best_policy[context] = 0
        
    def reset(self):
         for context in self.context_set:
            # for arm-value estimation
            self.nb_observation[context] = np.zeros(self.nbArm)
            self.accumulated_value[context] = np.zeros(self.nbArm)
            # the static game is formulated on arm_estimate
            self.arm_estimate[context] = np.zeros(self.nbArm)
            
            self.learning_state[context] = STATE_EXPLORE
            self.ptbd_arm_value[context] = np.zeros(self.nbArm) # perturbed arm values
        
            self.nb_state_visit[context] = np.zeros((4, self.nbArm))
            self.nb_state_aligned[context] = np.zeros(self.nbArm)
                        
            #set as a default 3-tuple: (mood, reference action, reference payoff = 0 or none-zero)
            self.current_state[context] = [STATE_DISCONTENT, 0] 
            self.reference_reward[context] = 0 # record the real reference reward of the state
            
            self.best_policy[context] = 0
        
        
    # --- functionalities
    def explore(self, context=None, time=None):
        """
        explore() only update when no collision occurs on the selected arm, see Eq. (5) of [Wang2019]
        will update the value in learn_arm_value()
        """
        assert self.learning_state[context] == STATE_EXPLORE, "learning state does not match"#debug
            
        self.selected_arm = np.random.randint(self.nbArm)
        
        return self.selected_arm
    
    def learn_arm_value(self, context=None, arm_values=None, collisions=None):
        # must be called after explore
        assert self.learning_state[context] == STATE_EXPLORE, "learning state does not match"#debug
        assert len(arm_values) == self.nbArm and len(collisions) == self.nbArm, "inputs are invalid."        
        assert collisions[self.selected_arm] != 0, "arm selection error."
        
        if collisions[self.selected_arm] == 1:
            armID = self.selected_arm
            self.nb_observation[context][armID] = self.nb_observation[context][armID] + 1 # obtain a new valid arm-value observation
            self.accumulated_value[context][armID] = self.accumulated_value[context][armID] + arm_values[armID]
            
            self.arm_estimate[context][armID] = self.accumulated_value[context][armID]  / self.nb_observation[context][armID]
        else:
            pass # do not update
            
        return self.arm_estimate[context]
    
    def set_internal_state(self, context=None, input_state=STATE_EXPLORE):
        # input_state: 0 --explore, 1 -- trial-and-error, 2 -- exploitation
        if input_state < STATE_EXPLORE or input_state > STATE_EXPLOIT:
            raise Exception("input state is invalid")
                
        if input_state == STATE_EXPLORE:
            pass
        elif input_state == STATE_LEARN:
            self.ptbd_arm_value[context][:] = 0
        elif input_state == STATE_EXPLOIT:
            # do it once for all
            self.get_best_policy(context)
        else:
            raise Exception("input is not valid.")
                            
        self.learning_state[context] = input_state
        
    
    def perturb_estimated_payoff(self, context=None, epoch=None):
        """
        The perturbation of estimated arm values guarantees that there is a unique social optimal equialibrium for the static game.
        See Proposition 3 in [Wang2019]
        """
        assert epoch is not None and epoch > 0, "the epoch index is invalid"
        
        #get a perturbation, which is only computed at the beginning of the learning phase in each each
        perturbation = np.random.random_sample(self.nbArm) * self.xi/epoch        
        assert len(perturbation) == self.nbArm, "the dimension of perturbation is invalid"
        
        self.ptbd_arm_value[context] = self.arm_estimate[context] + perturbation
#        self.init_tne_states(context)

        return self.ptbd_arm_value[context]

    def init_tne_states(self, context=None, starting_state=None):
        """
        We have 4 states: Content (C), Hopeful (H), Watchful (W) and Discontent (D).
        For each agent in a given context, the total # of local intermediate states is 4 * nbArm
        
        """
        # if we turn (1) on, in each exploration phase the learning algorithm will only use the outcomes of game play in this epoch.
        self.nb_state_visit[context] = np.zeros((4, self.nbArm)) # (1): tracks the frequency state visits
        self.nb_state_aligned[context] = np.zeros(self.nbArm)
        
        # set as a default 3-tuple: (mood=discontent, reference action (arm)=0, reference payoff = 0 or zero)
        if starting_state is None:
            self.current_state[context] = [STATE_DISCONTENT, 0]
            
            # reference_reward records the real reference reward of the state, 
            # initialization sets all players to select arm 0 so the reward is 0 due to collision
            self.reference_reward[context] = 0    
        else:
            self.current_state[context] = starting_state
            self.reference_reward[context] = 0
    
    def learn_policy(self, context=None, time=None):
        #note that here time is not used   
        assert context is not None, "context is not given" #debug
        assert self.learning_state[context] == STATE_LEARN, "learning state does not match" #debug 
        
        self.selected_arm = self.update_static_game_action(context, self.current_state[context])
        
        return self.selected_arm            
    
    def update_static_game_action(self, context=None, current_state=None):
        """
        Update action in the static game according to Eq.(9)
        """
        if current_state[0] == STATE_CONTENT: # if content
            #content, Eq. (9), experiment with prob. epsilon
            seed = np.random.random_sample()
            if seed > self.epsilon:
                action = current_state[1]
            else:
                remaining_actions = list(range(self.nbArm))
                remaining_actions.pop(current_state[1]) 
                action_id =  np.random.randint(self.nbArm - 1) 
                action = remaining_actions[action_id]
                assert action != current_state[1], "sampled action is invalid."
                                
#                print("player {} taking action arm {}".format(self.playerID, action)) #debug                

        elif current_state[0] == STATE_HOPEFUL or current_state[0] == STATE_WATCHFUL: # if hopeful or watchful
            #hopeful or watchful
            action = current_state[1] # do not change
        elif current_state[0] == STATE_DISCONTENT: # if discontent
            #discontent
            action = np.random.randint(self.nbArm)
            assert action >=0 and action < self.nbArm, "sampled action is invalid."
        else:
            raise Exception("the mood of the current state is invalid")
            
        return action
            
    def update_game_state(self, context, collisions):
        """
        Update the state of agent in the static game according to Alg. 2 in [Wang2019].
        Note that self.current_state[context] is in the form of (mood, arm, value)
        """
        current_reward = 0 # this is the reward of the static game
        if collisions[self.selected_arm] == 1:
            current_reward = self.ptbd_arm_value[context][self.selected_arm]        
        
        if self.current_state[context][0] == STATE_CONTENT:# if content
            # the current mood is content
            if self.selected_arm != self.current_state[context][1]:
                if current_reward > self.reference_reward[context]:
                    G_delta_u = (self.alpha21 * (current_reward - self.reference_reward[context]) + self.alpha22)                
                    threshold = self.epsilon ** G_delta_u
                    
                    #update according to Eq. (10) with probability                
                    sampled_result = np.random.choice([0, 1], size=None, p=[threshold, 1-threshold])
                    
                    if sampled_result == 0:                                  
                        self.current_state[context][1] = self.selected_arm #update reference action
                        self.reference_reward[context] = current_reward
                    else:
                        pass
                else:
                    pass
            else: # no experimenting
                if current_reward > self.reference_reward[context]:
                    self.current_state[context][0] = STATE_HOPEFUL # hopeful
                elif current_reward < self.reference_reward[context]:
                    self.current_state[context][0] = STATE_WATCHFUL # watchful
                else: # current_reward == self.reference_reward[context]:
                    pass # do nothing
                
        elif self.current_state[context][0] == STATE_HOPEFUL: # if hopeful
            if current_reward > self.reference_reward[context]:
                self.current_state[context][0] = STATE_CONTENT # set to content                
                self.reference_reward[context] = current_reward                
            elif current_reward == self.reference_reward[context]:
                self.current_state[context][0] = STATE_CONTENT
            else:# current_reward < self.reference_reward[context]:
                self.current_state[context][0] = STATE_WATCHFUL # set to watchful  
                
        elif self.current_state[context][0] == STATE_WATCHFUL: # if watchful
            if current_reward > self.reference_reward[context]:
                self.current_state[context][0] = STATE_HOPEFUL # set to hopeful                
            elif current_reward == self.reference_reward[context]:
                self.current_state[context][0] = STATE_CONTENT
            else:# current_reward < self.reference_reward[context]:
                self.current_state[context][0] = STATE_DISCONTENT # set to discontent
        
        elif self.current_state[context][0] ==  STATE_DISCONTENT:
            if current_reward == 0:
                pass# remain discontent, keep exploring
            else:                
                F_u =  self.alpha11 * current_reward + self.alpha12 # update with the probability in Eq. (11)                                              
                threshold = self.epsilon ** F_u 
                
                sampled_result = np.random.choice([0, 1], size=None, p=[threshold, 1-threshold])
                if sampled_result == 0:                    
                    self.current_state[context][0] = STATE_CONTENT
                    self.current_state[context][1] = self.selected_arm #update reference action
                    
                    self.reference_reward[context] = current_reward     
                else:
                    pass #stay with the same state
                        
        else:
            raise Exception("unexpected state.")
                
        #update the number of visited states
        id_mood = self.current_state[context][0]
        id_action = self.current_state[context][1]
        
        self.nb_state_visit[context][id_mood][id_action] = 1 + self.nb_state_visit[context][id_mood][id_action]
        
        if id_mood == STATE_CONTENT and self.reference_reward[context] == current_reward:
            self.nb_state_aligned[context][id_action] = 1 + self.nb_state_aligned[context][id_action]
        
    def exploit(self, context = None, time=None):
        assert context is not None, "context is None"
        assert self.learning_state[context] == STATE_EXPLOIT, "learning state does not match"
        assert time is not None, "time is None"
        
#        self.selected_arm = self.get_best_policy(context) # if turning this on, we'll compute the best policy each time
        
        self.selected_arm = self.best_policy[context]
        return self.selected_arm #return the action
        
    def get_best_policy(self, context = None):
        assert context is not None, "context is None"
        
        mat_frequency = self.nb_state_aligned[context] # only count the Content mood 
                
        id_max = np.argmax(mat_frequency) #over the remaining action/arm axis
        
        self.best_policy[context] = id_max
        
#        print("TnE - {}: Player {}: arm {}".format(context, self.playerID, id_max)) # debug
        
        return id_max
    
"""
Implemented based on the method proposed in the paper, [Bistritz2019]
"Game of Thrones: Fully Distributed Learning for Multi-Player Bandits", by Ilai Bistritz and Amir Leshem, 
NeurIPS2019 
"""
class GoTPlayer(Player): # with almost the same structure of TnE
    def __init__(self, param):        
        self.horizon = param["horizon"] if "horizon" in param.keys() else 0
                    
        #for arm of a specific context-player
        self.playerID = param["playerID"]
        self.nbArm = param["nbArm"]
        self.nbPlayer = param["nbPlayer"] # used for determining the probaibliy of intermediate state switching
        
        #used in Eq. (10) and Eq. (11) in [Wang2019]
        self.epsilon = param["epsilon"] 
        
        # Initialization
        self.nb_observation = np.zeros(self.nbArm)
        self.accumulated_value = np.zeros(self.nbArm)
        self.arm_estimate = np.zeros(self.nbArm)
        
        self.learning_state = STATE_EXPLORE

        self.selected_arm = 0
        self.nb_state_visit = np.zeros((2, self.nbArm))
        
        self.current_state = [STATE_DISCONTENT, 0]
        
        self.max_u = 1
        self.best_policy = 0
        
        # requirement from [Bistritz2019], the discrepancy of sum of maximum value and the social-optimal value
        self.c = 1.2 # this is an estimation
        self.pert_factor = self.c * self.nbPlayer
#        self.reference_reward = 0 # the current version of GoT doesn't need a reference reward        
        
    def reset(self):
        self.nb_observation = np.zeros(self.nbArm)
        self.accumulated_value = np.zeros(self.nbArm)
            
        # the static game is formulated on arm_estimate
        self.arm_estimate = np.zeros(self.nbArm)
            
        self.learning_state = STATE_EXPLORE         
        
        self.selected_arm = 0
        self.nb_state_visit = np.zeros((2, self.nbArm))
                        
        #set as a default 3-tuple: (mood, reference action, reference payoff = 0 or none-zero)
        self.current_state = [STATE_DISCONTENT, 0] 
#        self.reference_reward = 0 
        
        self.max_u = 1
        self.best_policy = 0
     
    # --- functionalities
    def explore(self, context = None, time = None):
        """
        we will update the estimated arm values in function learn_arm_value()
        context and time are not used for this version
        """
        assert self.learning_state == STATE_EXPLORE, "learning state does not match"#debug
            
        self.selected_arm = np.random.randint(self.nbArm)
        
        return self.selected_arm
    
    def learn_arm_value(self, context = None, arm_values = None, collisions = None):
        # must be called after explore
        assert self.learning_state == STATE_EXPLORE, "learning state does not match"#debug
        assert len(arm_values) == self.nbArm and len(collisions) == self.nbArm, "inputs are invalid"        
        assert collisions[self.selected_arm] != 0, "arm selection error"
        
        if collisions[self.selected_arm] == 1:
            armID = self.selected_arm
            self.nb_observation[armID] = self.nb_observation[armID] + 1 # obtain a new valid arm-value observation
            self.accumulated_value[armID] = self.accumulated_value[armID] + arm_values[armID]
            
            self.arm_estimate[armID] = self.accumulated_value[armID]  / self.nb_observation[armID]
        else:
            pass # do nothing
            
        return self.arm_estimate
    
    def set_internal_state(self, context=None, input_state=STATE_EXPLORE):
        # GoT does not use context information
        # input_state: 0 --explore, 1 -- trial-and-error, 2 -- exploitation
        if input_state < STATE_EXPLORE or input_state > STATE_EXPLOIT:
            raise Exception("input state is invalid")
                
        if input_state == STATE_EXPLORE:
            pass
        elif input_state == STATE_LEARN:
            pass
        elif input_state == STATE_EXPLOIT:
            self.get_best_policy() # calculate once far all
        else:
            raise Exception("input is not valid.")
                            
        self.learning_state = input_state
        
    def initalize_static_game(self, epoch=None, context=None):   
        """
        State initialization is done in init_got_states,
        this function is to be removed in the future
        """
        id_max_u = np.argmax(self.arm_estimate)
        
        self.max_u = self.arm_estimate[id_max_u]

#        print("id {} - max u {}".format(id_max_u, self.max_u))# debug
        
    def init_got_states(self, context=None, starting_state=None):
        """
        We have 2 states: Content (C) and Discontent (D).
        For each agent in each context, the total # of local intermediate state is 2 * nbArm
        
        
        starting_state is used for initializing the state at the beginnning of the epoch
        """
        # if we turn (1) on, in each exploration phase the learning algorithm will only use the outcomes of game play in this epoch.
        self.nb_state_visit = np.zeros((2, self.nbArm)) # (1): tracks the frequency of state visits
                
        if starting_state is None:
            # set as a default 3-tuple: (mood=discontent, reference action (arm)=0, reference payoff = 0 or zero)
            self.current_state = [STATE_DISCONTENT, 0]
            
            # reference_reward records the real reference reward of the state, 
            # initialization sets all players to select arm 0 so the reward is 0 due to collision
#            self.reference_reward = 0 
        else:
            self.current_state = starting_state
#            self.reference_reward = 0 # need to learn and update the reference reward for the new static game

    
    def learn_policy(self, context=None, time=None):
        #note that here time is not used   
        assert self.learning_state == STATE_LEARN, "learning state does not match" #debug 
        
        self.selected_arm = self.update_static_game_action(None, self.current_state)
        
        return self.selected_arm            
    
    
    def update_static_game_action(self, context=None, current_state=None):
        """
        Update action in the static game
        """
        if current_state[0] == STATE_CONTENT: # if content
            #content, Eq. (8) Alg.2 of [Bistritz2019], experiment with prob. epsilon
            tmp_factor = self.pert_factor # perturbation factor
            
            # sampling method 1
            prob_no_change = 1 - self.epsilon**(tmp_factor)
            prob_rand_action = self.epsilon**(tmp_factor) / (self.nbArm - 1)
            
            action_array = list(range(self.nbArm))
            prob_array = np.zeros(self.nbArm)
            prob_array[:] = prob_rand_action
            prob_array[current_state[1]] = prob_no_change
                        
            action = np.random.choice(action_array, size=None, p=prob_array)      
            
            # sampling method 2
#            seed = np.random.random_sample()
#            if seed <= 1 - self.epsilon**(tmp_factor):
#                # at content state a player does not experiment frequently
#                action = current_state[1]
#            else:
#                remaining_actions = list(range(self.nbArm))
#                remaining_actions.pop(current_state[1]) 
#                action_id = np.random.randint(self.nbArm - 1) 
#                action = remaining_actions[action_id]
#                assert action != current_state[1], "sampled action is invalid."
                                
        elif current_state[0] == STATE_DISCONTENT: # if discontent
            #discontent
            action = np.random.randint(self.nbArm)
            assert action >=0 and action < self.nbArm, "sampled action is invalid."
        else:
            raise Exception("the mood of the current state is invalid")
            
        return action

    def update_game_state(self, context, collisions, flag_record_frequency=False):
        """
        Ignore any context. The GoT algorithm is designed for the MP-MAB in stochastic environment w/o context
        """
        current_reward = 0 # this is the reward of the static game
        if collisions[self.selected_arm] == 1:
            current_reward = self.arm_estimate[self.selected_arm]
        elif collisions[self.selected_arm] == 0:
            raise Exception("the collision is not correctly computed.") 
        else:
            current_reward = 0 # if there is a collision
        
        if self.current_state[0] == STATE_CONTENT:# if content
            # the current mood is content
            # check the current reward first
            if current_reward <= 0:
                self.current_state[0] = STATE_DISCONTENT
                self.current_state[1] = self.selected_arm
            else:
                # current_reward > 0
                if self.selected_arm == self.current_state[1]:
                    # If the current action is the same as the reference action,
                    # and utility > 0, then a content player remains content with probability 1
                    pass # stay at the same state, w/ probability 1
                elif self.selected_arm != self.current_state[1]:
                    # set the probability
                    threshold = current_reward / self.max_u * (self.epsilon**(self.max_u - current_reward))
                    sampled_result = np.random.choice([0, 1], size=None, p=[threshold, 1-threshold])      
                 
                    if sampled_result == 0:
                        self.current_state[0] = STATE_CONTENT
                        self.current_state[1] = self.selected_arm       
                        
#                        info_logger().log_info('Player {}: action {} remains CONTENT with prob. {}'.format(self.playerID, self.selected_arm, threshold)) #debug                       
                    else:
                        self.current_state[0] = STATE_DISCONTENT
                        self.current_state[1] = self.selected_arm       
                        
#                        info_logger().log_info('Player {}: action {} transit to DISCONTENT with prob. {}'.format(self.playerID, self.selected_arm, threshold))#debug 
        
        elif self.current_state[0] == STATE_DISCONTENT:
            if current_reward <= 0:
                self.current_state[0] = STATE_DISCONTENT
                self.current_state[1] = self.selected_arm
            else:                
                threshold = current_reward / self.max_u * (self.epsilon**(self.max_u - current_reward))
                sampled_result = np.random.choice([0, 1], size=None, p=[threshold, 1-threshold])
                                 
                if sampled_result == 0:
                    self.current_state[0] = STATE_CONTENT
                    self.current_state[1] = self.selected_arm
                    
#                    info_logger().log_info('Player {}: action {} transit to CONTENT with prob. {}'.format(self.playerID, self.selected_arm, threshold)) #debug
                else:
                    self.current_state[0] = STATE_DISCONTENT
                    self.current_state[1] = self.selected_arm
        else:
            raise Exception("unexpected state.")

        # only the last few rounds are considered to count toward the optimal policy
        if flag_record_frequency == True:                
            #update the number of visited states
            id_mood = 0 if self.current_state[0] == STATE_CONTENT else 1
            id_action = self.current_state[1]            

            self.nb_state_visit[id_mood][id_action] = 1 + self.nb_state_visit[id_mood][id_action]
        
    def exploit(self, context = None, time=None):
        assert time is not None, "time is None"
        assert self.learning_state == STATE_EXPLOIT, "learning state does not match at iteration {}".format(time)
                
#        self.selected_arm = self.get_best_policy(context) # if turning this line on, we'll compute the best policy each time
        
        self.selected_arm = self.best_policy
        return self.selected_arm #return the action
        
    def get_best_policy(self, context = None):       
        mat_frequency = self.nb_state_visit[0,:] # over the mood axis, over CONTENT
        assert np.shape(mat_frequency) == (self.nbArm,), "shape of frequency is wrong."
                
        id_max = np.argmax(mat_frequency) #over the remaining action/arm axis
        
        self.best_policy = id_max
        
#        info_logger().log_info("GoT - Player {}: frequency {} arm {}".format(self.playerID, mat_frequency, id_max)) #debug        
        
        return id_max