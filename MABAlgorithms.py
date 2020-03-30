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
# 1. Hungarian: standard Hungarian algorithm for centralized arm allocation,
# 2. StaticHungarian: centralized allocation algorithm if the mean value of each arm is known
# 3. MusicalChairs: Musical Chairs algorithm for multi-player, homogeneous multi-arm bandit
# 4. TrialandError: Log-linear learning algorithm for contextual multi-player multi-arm bandits,
#    with heterogeneous arms
# 5. GameofThrone: Log-linear learning algorithm for multi-player multi-arm bandits with heterogeneous 
#    arms. It is sub-optimal for contextual bandits
#
# Typically, one distributed algorithm is accompanied by a corresponding player class
# see also MABAlgorithms2.py

__author__ = "Wenbo Wang"

import numpy as np
from scipy.optimize import linear_sum_assignment
from Players import MusicChairPlayer, TnEPlayer, GoTPlayer

from loggingutils import info_logger

if __name__ == '__main__':
    print("Warning: this script 'MABAlgorithms.py' is NOT executable..")  # DEBUG
    exit(0)


class MABAlgorithm(object):
    """ Base class for an algorithm class."""
    def __init__(self, param):
        """ Base class for an algorithm class."""
        self.nbPlayer = param["nbPlayer"]
        self.nbArm = param["nbArm"]
        self.context_set = param["context_set"]
        
        self.nbAgent = 0 # number of agents in the algorithms, can be centralized, decentralized or partially decentralized
        
        # an agent is usually corresponding to a player, it has its own 
        self.agents = []
        
    # --- Printing
    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.__dir__)

    # --- functionalities
    def resolve_collision(self, pulls):
        (nbPlayer, nbArm) = np.shape(pulls)
        assert nbPlayer == self.nbPlayer and nbArm == self.nbArm, "input does not match the stored environment parameters."
        assert nbPlayer <= nbArm, "player number should be larger than or equal to arm number."
    
        collisions = pulls.sum(axis=0)
        
        assert len(collisions) == nbArm, "dimension of collisions is incorrect"
        return collisions   
        
    def learn_policy(self, game_env, context=None, time=None):
        """
        Learn policies based on the given game environments.
        A game environment can be in the form of (context, sampel_reward_matrix)
        """
        raise NotImplementedError("This method learn_policy(t) has to be implemented in the class inheriting from MABAlgorithm.") 

    def reset(self, horizon=None):
        """
        The rest parameters cannot be reset, except self.horizon.
        """
        raise NotImplementedError("This method reset() has to be implemented in the class inheriting from MABAlgorithm.") 

    def pulls2choices(self, pulls):
        """
        Convert pulls into choices
        """        
        (nbPlayer, nbArm) = np.shape(pulls)
        assert nbPlayer == self.nbPlayer and nbArm == self.nbArm, "input does not match the stored environment parameters."
        
        arm_choices = np.zeros(nbPlayer, dtype=int)
        
        arm_selected = np.nonzero(pulls) # index of non-zero values
        
        # for some algorithms there may be a case when a player refuse to choose any arm    
        for index in range(len(arm_selected[0])):
            playerID = arm_selected[0][index]
            arm_choices[playerID] = arm_selected[1][index] # playerID should be 0, 1, 2,..., nbPlayer-1
           
        return arm_choices
    
    def observe_distributed_payoff(self, game_env, collisions):
        (nbPlayer, nbArm) = np.shape(game_env)
        assert nbPlayer == self.nbPlayer and nbArm == self.nbArm, "input does not match the stored environment parameters."
        
        current_reward = np.zeros(self.nbPlayer)
        
        for playerID in range(self.nbPlayer):
            selected_arm = self.agents[playerID].selected_arm
            
            # for some algorithms there may be a case when a player refuses to choose any arm    
            if selected_arm < 0:
                current_reward[playerID] = 0
            else:
                if collisions[selected_arm] == 1:
                    current_reward[playerID] = game_env[playerID][selected_arm]# not collidiing
                else:
                    current_reward[playerID] = 0# colliding or void
    
        # returen an array of dimension nbArm
        return current_reward        

"""
 Algorithm: centralized Hungarian
"""    
class Hungarian(MABAlgorithm):
    """
    Centralized assignment algorithm in the form of Hungarian (Munkres) algorithm. 
    Implemented based on scipy.optimize.linear_sum_assignment.
    It does not have the structure of multiple agents as the other algorithms.
    """
    def __init__(self, param):
        self.nbPlayer = param["nbPlayer"]
        self.nbArm = param["nbArm"]
        self.context_set = param["context_set"]
        """For simplicity we do not implement the single agent here."""
#        self.nbAgent = 0
        self.agents = []

    # --- Printing
    def __str__(self):
        return "Hungarian"

    # --- functionalities
    def reset(self, horizon=None):
        pass # do nothing

    def learn_policy(self, game_env, context=None, time=None):
        # context is not used in Hungarian
        (nbPlayer, nbArm) = np.shape(game_env)
        assert nbPlayer == self.nbPlayer and nbArm == self.nbArm, "input does not match the stored environment parameters."
        assert nbPlayer <= nbArm, "player number should be larger than or equal to arm number."
                        
        #the mehtod requires the number of rows (jobs) to be larger than that of columns (workers)
        cost_matrix = np.negative(game_env.transpose())
        # note that the cost_matrix is a transpose of the original matrix
        col_ind, row_ind = linear_sum_assignment(cost_matrix) 
        
        pulls = np.zeros((nbPlayer, nbArm))
        sampled_rewards = np.zeros(nbPlayer)
        for ii in range(len(row_ind)):
            playerID = row_ind[ii]
            sampled_rewards[playerID] = game_env[playerID][col_ind[ii]]
            pulls[playerID, col_ind[ii]] = 1            
        
        total_rewards = game_env[row_ind, col_ind].sum()
        
        return pulls, total_rewards, sampled_rewards
    
"""
 Algorithm: centralized Hungarian over means of arm-values (static values)
"""  
class StaticHungarian(Hungarian):
    """
    This algorithm is implemented for the purpose of deriving the throetic regret
    """
    def __init__(self, param):
        super().__init__(param)         
        self.pulls = {}
        
        #we keep them for later use
        self.total_rewards = {}
        self.static_rewards = {}
        
        self.mean_env_payoff = param["mean_game_env"]
        self.flag_allocation_ready = False
        
        for context in self.context_set:
            self.pulls[context] = np.zeros((self.nbPlayer, self.nbArm))
            self.total_rewards[context] = 0
            self.static_rewards[context] = np.zeros(self.nbPlayer)
        
        self.array_context = param["array_context"]
        self.array_prob = param["array_prob"]
        
        self.mean_total_reward = 0
        self.mean_static_reward = np.zeros(self.nbPlayer)
        
    # --- Printing
    def __str__(self):
        return "Static Hungarian"
         
    def reset(self, horizon=None):
        self.mean_total_reward = 0
        self.mean_static_reward = np.zeros(self.nbPlayer)
        self.flag_allocation_ready = False
         
    def learn_policy(self, game_env, context=None, time=None):
        #ignore all the inputs        
        if self.flag_allocation_ready == False:     
            for context_id in range(len(self.array_context)):
                tmp_context = self.array_context[context_id]
                self.pulls[tmp_context], self.total_rewards[tmp_context], self.static_rewards[tmp_context] = super().learn_policy(
                        self.mean_env_payoff[tmp_context], tmp_context)
                
                self.mean_total_reward = self.mean_total_reward + self.total_rewards[tmp_context] * self.array_prob[context_id]
                self.mean_static_reward = self.mean_static_reward + self.static_rewards[tmp_context] * self.array_prob[context_id]
                
#                print("Static Hungarian: {}".format(tmp_context))
            
            self.flag_allocation_ready = True

        return self.pulls[context], self.mean_total_reward, self.mean_static_reward
    
"""
 Algorithm: musical chairs
"""          
class MusicalChairs(MABAlgorithm):
    """
    Decentralized assignment algorithm in the form of Musical Chair algorithm. 
    Implemented based on the paper "Multi-Player Bandits – a Musical Chairs Approach", by Jonathan Rosenski and 
    Ohad Shamir @2015 [Rosenski2015]. Note that this algorithm is designed for multi-player only and for 
    contextual bandit it adapts to the condition of unobservable context.
    """
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
            
            self.agents.append(MusicChairPlayer(player_param))
        
        self.time = 0
        self.T0 = self.agents[0].T0
        
    # --- Printing
    def __str__(self):
        return "Musical Chairs"
    
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
        
        if time <= self.T0:
            for agentID in range(nbPlayer):
                armID = self.agents[agentID].explore(context, time)
                pulls[agentID][armID] = 1  
                
            collisions = self.resolve_collision(pulls)
        
            for agentID in range(nbPlayer):
                self.agents[agentID].learn_arm_value(context, game_env[agentID,:], collisions)
        else:
            for agentID in range(nbPlayer):
                armID = self.agents[agentID].exploit(context, time)
                pulls[agentID][armID] = 1  
        
            collisions = self.resolve_collision(pulls)
             
            for agentID in range(nbPlayer):
                self.agents[agentID].update_musical_chair(time, collisions)
        
        sampled_rewards = self.observe_distributed_payoff(game_env, collisions)            
        total_rewards = np.sum(sampled_rewards)        
        return pulls, total_rewards, sampled_rewards
    
"""
 Algorithm: trial and error [Wang2019]
"""   
class TrialandError(MABAlgorithm):
    """
    Decentralized assignment algorithm in the form of trial-and-error learning algorithm. 
    Implemented for the paper "Decentralized Learning for Channel Allocation in IoT Networks over Unlicensed Bandwidth as a 
    Contextual Multi-player Multi-armed Bandit Game", by Wenbo Wang et al.
    Note that this algorithm is designed for multi-player when contextual information is observable.
    (If context is not observable, the algorithm produces a sub-optimal allocation in the same level as a distributed learning 
    algorithm for non-contextual MP-MAB)
    """
    def __init__(self, param):
        self.nbPlayer = param["nbPlayer"]
        self.nbArm = param["nbArm"]
        self.context_set = param["context_set"]
        self.horizon = param["horizon"] # agents don't know the fixed horizon when running the algorithm
        
        # each player will be attached a single agent
#        self.nbAgent = self.nbPlayer
        self.agents = []
        
        self.xi = param["xi"] if "xi" in param.keys() else 0.001
        
        # a large epsilon will leads to more frequent transtions (explorations) in the intermedate game
        self.epsilon = param["epsilon"] if "epsilon" in param.keys() else 0.1         
        # see Theorem 1 in [Wang2019], not kept by the agents, determining trial-and-error rounds
        self.delta = param["delta"] if "delta" in param.keys() else 2
        
        self.rho = param["rho"] if "rho" in param.keys() else 0.5 # no longer used by the improved algorithm
        
        self.exploration_round = param["c1"]
        self.c2 = param["c2"]
        self.c3 = param["c3"]
        
        for playerID in range(self.nbPlayer):
            player_param = {"context_set": self.context_set, 
                            "nbArm": self.nbArm,
                            "playerID": playerID,
                            "xi": self.xi,
                            "epsilon": self.epsilon,
                            "delta": self.delta,
                            "rho": self.rho,
                            "alpha11": param['alpha11'] if 'alpha11' in param.keys() else None,
                            "alpha21": param['alpha21'] if 'alpha21' in param.keys() else None,
                            "alpha12": param['alpha12'] if 'alpha12' in param.keys() else None,
                            "alpha22": param['alpha22'] if 'alpha22' in param.keys() else None
                            }
            
            self.agents.append(TnEPlayer(player_param))
        
        self.time = 0        
        # used for determining the epoch
        self.epoch = 1
        
        #initialize for the first epoch
        self.tne_round = self.exploration_round + self.c2 # *1
        self.rounds_in_epoch = self.tne_round + self.c3*2 # * (2** 1) # rounds in the first epoch
        self.current_round = 1 
        
        self.flag_observable = True # set if the context is observable
        
        # for debug purpose
        self.nbExploration = 0
        self.nbTnE = 0
        self.nbExploitation = 0
        
    # --- Printing
    def __str__(self):
        return "Trial and Error"    
    
    def set_context_observability(self, flag_observable = True):
        """
        set_context_observability() turns on/off the observability of contexts (side information), 
        see Section V. of [Wang2019]. 
        """
        self.flag_observable = flag_observable
    
    # --- functionalitiess        
    def reset(self, horizon=None):
        for agent in self.agents:
            agent.reset()
            
        self.time = 0
        self.epoch = 1
      
        # reset to the initial values
        self.tne_round = self.exploration_round + self.c2 # *1
        self.rounds_in_epoch = self.tne_round + self.c3*2 # * (2** 1) # rounds in the first epoch
        self.current_round = 1 
        
        self.nbExploration = 0
        self.nbTnE = 0
        self.nbExploitation = 0
        
        if horizon is not None:
            self.horizon = horizon
    
    def learn_policy(self, game_env, context=None, time=None):
        """
        learn_policy() implements the 3 phases in Alg. 1 of [Wang2019]. 
        """
        (nbPlayer, nbArm) = np.shape(game_env)
        assert nbPlayer == self.nbPlayer and nbArm == self.nbArm, "input does not match the stored environment parameters."
        assert nbPlayer <= nbArm, "player number should be larger than or equal to arm number."
        assert time is not None, "time is not given."
           
        if self.flag_observable == False:
            # freeze the context s.t. the algorithm is reduced to an MP-MAP
            context = list(self.context_set)[0] 
            
        self.time = self.time + 1
        
        if self.current_round > self.rounds_in_epoch:
            #update epcoh
            self.epoch = self.epoch + 1
            # rounds in the k-th epoch
            self.tne_round = int(self.exploration_round + self.c2*(self.epoch**self.delta)) # insce delta may be non-integer
            self.rounds_in_epoch = int(self.tne_round + self.c3*(2**self.epoch))
            #reset
            self.current_round = 1
#            print("number of epoch: {}".format(self.epoch))# debug
        
        pulls = np.zeros((nbPlayer, nbArm))
        
        if self.current_round <= self.exploration_round:# exploration rounds
            # reset the phase to exploration in an epoch
            if self.current_round == 1:
                for agentID in range(nbPlayer):
                    for tmp_context in self.context_set:
                        self.agents[agentID].set_internal_state(tmp_context, 0)         

#                print("reset iteration at epoch {}".format(self.epoch))# debug            

            #exploration by randomly choosing actions
            for agentID in range(nbPlayer):
                armID = self.agents[agentID].explore(context, time)
                pulls[agentID][armID] = 1         
                
            collisions = self.resolve_collision(pulls)        
            for agentID in range(nbPlayer):
                self.agents[agentID].learn_arm_value(context, game_env[agentID,:], collisions)
                
            current_rewards = self.observe_distributed_payoff(game_env, collisions)
            
            # for debugging
            self.nbExploration = self.nbExploration + 1
                
        elif self.current_round <= self.tne_round:# trial-and-error phase
            if self.current_round == self.exploration_round + 1:
                # reset the phase to learning in an epoch
                for agentID in range(nbPlayer):
                    for tmp_context in self.context_set:
                        self.agents[agentID].set_internal_state(tmp_context, 1) 
                        #set the static game
                        self.agents[agentID].perturb_estimated_payoff(tmp_context, self.epoch)   
                        
                        # get the latest best policy (from the last epoch)
                        init_state = None
                        if self.epoch != 1:
                            init_state = [0, self.agents[agentID].best_policy[tmp_context]]
                        else:
                            #randomize
                            action = np.random.randint(self.nbArm)
                            init_state = [0, action]
                            
                        # can be moved into perturb_estimated_payoff() in the later versions
                        self.agents[agentID].init_tne_states(tmp_context, init_state) 
                                            
            #trial-and-error phase, taking actions randomly according to the intermediate state
            for agentID in range(nbPlayer):
                armID = self.agents[agentID].learn_policy(context)
                pulls[agentID][armID] = 1  

            collisions = self.resolve_collision(pulls)         
            
            for agentID in range(nbPlayer):
                self.agents[agentID].update_game_state(context, collisions)
            
            #update reward according to actions taken
            current_rewards = self.observe_distributed_payoff(game_env, collisions)
                
            # for debugging
            self.nbTnE = self.nbTnE + 1
        else:
            if self.current_round == self.tne_round + 1:
                 # reset the phase to exploration in an epoch
                for agentID in range(nbPlayer):
                    for tmp_context in self.context_set:
                        self.agents[agentID].set_internal_state(tmp_context, 2)  
                        
                ###############################################################
                # Debugging
                for agentID in range(nbPlayer):
                    armID = self.agents[agentID].exploit(context, self.current_round)
                    pulls[agentID][armID] = 1 
                collisions = self.resolve_collision(pulls)
                    
                info_logger().log_info('TnE Context {}: collisions array {}'.format(context, collisions)) #debug
                # End of debugging
                ###############################################################
            
            #exploitation
            for agentID in range(nbPlayer):
                armID = self.agents[agentID].exploit(context, self.current_round)
                pulls[agentID][armID] = 1  
                
            collisions = self.resolve_collision(pulls)              
            current_rewards = self.observe_distributed_payoff(game_env, collisions)
            
            # for debugging
            self.nbExploitation = self.nbExploitation + 1
        
        #update round number
        self.current_round = self.current_round + 1
            
        total_rewards = np.sum(current_rewards)        
        return pulls, total_rewards, current_rewards
        
"""
 Algorithm: trial and error [Leshem2018]
"""   
class GameofThrone(MABAlgorithm):    
    """
    Decentralized assignment algorithm in the form of game-of-throne learning algorithm. 
    Implemented for the paper "Distributed Multi-Player Bandits - a Game of Thrones Approach", by Ilai Bistritz et al.
    Note that this algorithm is designed for multi-player without considering contextual information.
    """
    def __init__(self, param):
        self.nbPlayer = param["nbPlayer"]
        self.nbArm = param["nbArm"]
        self.horizon = param["horizon"] # agents don't know the fixed horizon when running the algorithm
        
        # each player will be attached a single agent
#        self.nbAgent = self.nbPlayer
        self.agents = []
        
        # a large epsilon will leads to more frequent transtions (explorations) in the intermedate game
        self.epsilon = param["epsilon"] if "epsilon" in param.keys() else 0.1         
        # see Theorem 1 in [Wang2019], not kept by the agents, determining trial-and-error rounds
        self.delta = param["delta"] if "delta" in param.keys() else 2  
        # set the round of iteration where we 
        self.rho = param["rho"] if "rho" in param.keys() else 0.5
        
        self.c1 = param["c1"]        
        self.c2 = param["c2"]
        self.c3 = param["c3"]
        
        for playerID in range(self.nbPlayer):
            player_param = {"nbArm": self.nbArm,
                            "nbPlayer": self.nbPlayer,
                            "playerID": playerID,
                            "epsilon": self.epsilon,
                            "delta": self.delta
                            }
            
            self.agents.append(GoTPlayer(player_param))
        
        self.time = 0    
        # used for determining the epoch
        self.epoch = 1
        
        # initialize for the first epoch, 
        # for simplicity, the parameter names are kept the same as the TnE algorithm.
        self.exploration_round = self.c1
        self.got_round = self.exploration_round + self.c2 # *1
        self.rounds_in_epoch = self.got_round + self.c3*2 # * (2** 1) # rounds in the first epoch
        self.current_round = 1                     
        
    # --- Printing
    def __str__(self):
        return "Game of Throne"    
    
    # --- functionalitiess     
    def reset(self, horizon=None):
        for agent in self.agents:
            agent.reset()
            
        self.time = 0
        self.epoch = 1
      
        # reset to the initial values
        self.got_round = self.exploration_round + self.c2 # *1
        self.rounds_in_epoch = self.got_round + self.c3*2 # * (2** 1) # rounds in the first epoch
        self.current_round = 1         
        
        if horizon is not None:
            self.horizon = horizon

    def learn_policy(self, game_env, context=None, time=None):
        """
        learn_policy() implements the 3 phases in Alg. 1 of [Leshem2018]. 
        Implemented in the same structure for tial-and-error
        """
        (nbPlayer, nbArm) = np.shape(game_env)
        assert nbPlayer == self.nbPlayer and nbArm == self.nbArm, "input does not match the stored environment parameters."
        assert nbPlayer <= nbArm, "player number should be larger than or equal to arm number."
        assert time is not None, "time is not given."
            
        self.time = self.time + 1
        
        if self.current_round > self.rounds_in_epoch:
            #update epcoh
            self.epoch = self.epoch + 1
            # rounds in the k-th epoch
            self.exploration_round = int(self.c1*(self.epoch**self.delta))
            self.got_round = int(self.exploration_round + self.c2*(self.epoch**self.delta))
            self.rounds_in_epoch = int(self.got_round + self.c3*(2**self.epoch))
            #reset
            self.current_round = 1
#            print("number of epoch: {}".format(self.epoch))# debug
        
        pulls = np.zeros((nbPlayer, nbArm))
        
        if self.current_round <= self.exploration_round:# exploration rounds
            # reset the phase to exploration in an epoch
            if self.current_round == 1:
                for agentID in range(nbPlayer):
                    self.agents[agentID].set_internal_state(context=None, input_state=0)                

            # exploration by randomly choosing actions
            for agentID in range(nbPlayer):
                armID = self.agents[agentID].explore(None, time)
                pulls[agentID][armID] = 1         
                
            collisions = self.resolve_collision(pulls)        
            for agentID in range(nbPlayer):
                self.agents[agentID].learn_arm_value(None, game_env[agentID,:], collisions)
                
            # learn the real payoff
            current_rewards = self.observe_distributed_payoff(game_env, collisions)            
                
        elif self.current_round <= self.got_round:# game-and-thrones phase
            if self.current_round == self.exploration_round + 1:
                # reset the phase to learning in an epoch
                for agentID in range(nbPlayer):                   
                    self.agents[agentID].set_internal_state(context=None, input_state=1) 
                    
                    # as per Alg.1 in [Leshem2018], initialize the mood to be content
                    if self.epoch != 1:
                        init_state = [0, self.agents[agentID].best_policy] #(STATE_CONTENT, BEST ACTION)
                    else:
                        #randomize
                        action = np.random.randint(self.nbArm)
                        init_state = [0, action]
                         
                    # initialize the intermediate game
                    self.agents[agentID].initalize_static_game(init_state, self.epoch)   
                    # initialize the intermediate states, and (TODO) this can be moved into perturb_estimated_payoff() 
                    self.agents[agentID].init_got_states(context=None, starting_state=init_state) 
        
            #game of throne phase, taking actions randomly according to the intermediate state
            for agentID in range(nbPlayer):
                armID = self.agents[agentID].learn_policy(context=None)
                pulls[agentID][armID] = 1  

            collisions = self.resolve_collision(pulls)         
            
            flag_count_frequency = False
            # update the count of state-visit only for the last half starting from rho*c2*k^(1+delta) rounds
#            if self.current_round >= self.got_round - 1 - self.rho*self.c2*(self.epoch**self.delta):
            if self.current_round >= self.exploration_round + self.rho*self.c2*(self.epoch**self.delta):
                flag_count_frequency = True
#            flag_count_frequency = True

            for agentID in range(nbPlayer):
                self.agents[agentID].update_game_state(context=None, collisions=collisions, 
                           flag_record_frequency=flag_count_frequency)
            
            #update reward according to actions taken
            current_rewards = self.observe_distributed_payoff(game_env, collisions)
                
        else:
            if self.current_round == self.got_round + 1:
                # reset the phase to exploitation in an epoch
                for agentID in range(nbPlayer):   
                    # the best policy is computed in set_internal_state()    
                    self.agents[agentID].set_internal_state(context=None, input_state=2)                        
                
                ###############################################################
                # Debugging
                for agentID in range(nbPlayer):
                    armID = self.agents[agentID].exploit(None, self.current_round)
                    pulls[agentID][armID] = 1 
                collisions = self.resolve_collision(pulls)
                    
                info_logger().log_info('GoT Context {}: collisions array {}'.format(context, collisions)) #debug
                # End of debugging
                ###############################################################
            
            #exploitation
            for agentID in range(nbPlayer):
                armID = self.agents[agentID].exploit(None, self.current_round)
                pulls[agentID][armID] = 1                  
                
            collisions = self.resolve_collision(pulls)              
            current_rewards = self.observe_distributed_payoff(game_env, collisions)
        
        #update round number
        self.current_round = self.current_round + 1
            
        total_rewards = np.sum(current_rewards)        
        return pulls, total_rewards, current_rewards

__all__ = ["Hungarian", "StaticHungarian", "MusicalChairs", "TrialandError", "GameofThrone"]