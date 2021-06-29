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

# This file defines the evaluation and virtualization mechanisms of the simulations.
# class AlgEvaluator
#
# For each simulation there are two versions: single-process and multi-process (parallel).
# Note that the parallel version is usually 2X to 4X faster than the single-process version, depending on the 
# simulation configuration. However, it is at the cost of consuming the same folds of memory. 
# It may use up the machine memory and result in a program collapse when the horizon/player nunber/arm number
# is too large

__author__ = "Wenbo Wang"

from tqdm import tqdm
import multiprocessing as mp
import numpy as np

#import environemnt generators
from MPMAB import MP_MAB
from HetNetSimulator import HomeBrewedHetNetEnv

#import algorithms
from MABAlgorithms import Hungarian, StaticHungarian, MusicalChairs, TrialandError, GameofThrone
from MABAlgorithms2 import SOC
from MABAlgorithms3 import SICMMAB

from MABAlgorithms2a import ESE

from loggingutils import info_logger

# result recorder
from PlayResult import ResultMultiPlayers

if __name__ == '__main__':
    print("Warning: this script 'GameEvaluator.py' is NOT executable..")  # DEBUG
    exit(0)

class AlgEvaluator:
    def __init__(self, configuration):        
        self.horizon = configuration['horizon']
        
        self.nbArms = configuration['arm number']        
        self.nbPlayers = configuration['player number']
        
        self.context_set = configuration['context set']
        self.nbContext = len(self.context_set)
        
        # for loaded values or when calling the prepare() methods, set true
        self.flag_pre_prepare = False        
        self.flag_simulation_done = False
                
        # we only have a unique bandit game, but may have more than one algorithms
        self.mp_mab_env = None
        # to be extended
        if configuration['env_type'] == 'uniform':            
            self.mp_mab_env = MP_MAB.uniform_mab(self.context_set, self.nbArms, self.nbPlayers, 
                                                  dic_lower = configuration['initial data'][0], 
                                                  dic_upper = configuration['initial data'][1])
            
            # 'context probabilites' is used for a differernt purpose in HetNet simulator
            if 'context probabilites' in configuration.keys():
                # set arbitrary probabilities for discrete context distribution
                context_probabilites = configuration['context probabilites']
                self.mp_mab_env.set_discrete_context_prob(context_probabilites)
        elif configuration['env_type'] == 'gaussian':
            self.mp_mab_env = MP_MAB.uniform_mab(self.context_set, self.nbArms, self.nbPlayers, 
                                                  dic_mean = configuration['initial data'][0], 
                                                  dic_sigma = configuration['initial data'][1])
            
            # 'context probabilites' is used for a differernt purpose in HetNet simulator
            if 'context probabilites' in configuration.keys():
                # set arbitrary probabilities for discrete context distribution
                context_probabilites = configuration['context probabilites']
                self.mp_mab_env.set_discrete_context_prob(context_probabilites)       
        elif configuration['env_type'] == 'HetNet simulator':
            hetnet_params = {'enabel mmWave': configuration['enabel mmWave'],
                             'horizon': self.horizon,
                             'cell range': configuration['cell range'],
                             'context_prob': configuration['context_prob'],
                             'los_prob': configuration['los_prob']
                             }
            
            self.mp_mab_env = HomeBrewedHetNetEnv.HetNet_mab(self.context_set, self.nbArms, self.nbPlayers, 
                                                             hetnet_params)            
#            print("showing UE and MUE positions") #debugging
#            self.mp_mab_env.helper_plot_ue_posiiton() #debugging
            
        elif configuration['env_type'] == 'load data':
            #TODO: load the series of arm values from a existing file
#            self.flag_pre_prepare = True
            pass        
                        
        self.algorithms = [] # a list of algorithms        
        self.result_recorders = [] # a list of result recorder for each algorithm  
        self.alg_names = []
        
    def prepare_arm_samples(self, horizon = None):
        if horizon is not None:
            self.horizon = horizon
        
        self.mp_mab_env.prepare_samples(self.horizon)
            
        self.flag_pre_prepare = True
        
    def reset_player_number(self, nbPlayer=None):
        # it is allowed only to be done after the samples are prepared
        if nbPlayer is None or self.flag_pre_prepare == False:
            return False
        else:
            self.nbPlayers = nbPlayer
            self.mp_mab_env.nbPlayers = nbPlayer
            
            return True
            
    def reset_arm_number(self, nbArm=None):
        # it is allowed only be done after the samples are prepared
        # we are not goning to change the real record of the arm values
        if nbArm is None or self.flag_pre_prepare == False:
            return False
        else:
            self.nbArms = nbArm
            self.mp_mab_env.nbArms = nbArm
            
            return True
    
    def clear_algorithms(self):
        # clear all existing algorithms and their corresponding recorders
        self.algorithms = []
        self.result_recorders = []
        self.alg_names = []
    
    def add_algorithm(self, algo_type = 'Trial and Error', custome_params=None):
        """ Create environments."""
        alg_params = {"nbPlayer": self.nbPlayers, "nbArm": self.nbArms, "context_set": self.context_set}
        
        #for each algorithm, append a recorder
        if algo_type == 'Trial and Error' or algo_type == 'TnE Nonobservable':
            #create a trial-and-error algorithm
            alg_params["horizon"] = self.horizon
            alg_params["c1"] = custome_params["c1"] if custome_params is not None else 100
            alg_params["c2"] = custome_params["c2"] if custome_params is not None else 5
            alg_params["c3"] = custome_params["c3"] if custome_params is not None else 1
            
            alg_params["epsilon"] = custome_params["epsilon"] if custome_params is not None else 0.1
            alg_params["delta"] = custome_params["delta"] if custome_params is not None else 2
                    
            if "alpha11" in custome_params.keys():
                alg_params["alpha11"] = custome_params["alpha11"]
            
            if "alpha12" in custome_params.keys():
                alg_params["alpha12"] = custome_params["alpha12"]
                
            if "alpha21" in custome_params.keys():
                alg_params["alpha21"] = custome_params["alpha21"]
                
            if "alpha22" in custome_params.keys():
                alg_params["alpha22"] = custome_params["alpha22"]                
            
            alg_TnE = TrialandError(alg_params)
                        
            if  "observable" in custome_params.keys():
                alg_TnE.set_context_observability(custome_params["observable"]==1)
            
            self.algorithms.append(alg_TnE)
                        
            if algo_type == 'Trial and Error':
                result_TnE = ResultMultiPlayers(algo_type, 
                                            self.context_set, self.nbPlayers, self.nbArms, self.horizon) 
                self.result_recorders.append(result_TnE)
                self.alg_names.append(algo_type)
            else:
                result_TnE = ResultMultiPlayers('Non-Contextual TnE', 
                                            self.context_set, self.nbPlayers, self.nbArms, self.horizon)  
                self.result_recorders.append(result_TnE)                                              
                self.alg_names.append('Non-Contextual TnE')
            
        elif algo_type == 'Musical Chairs': #str(MusicalChair)
            alg_params["horizon"] = self.horizon
            # 3000 is hardcoded, as given by the original paper [Rosenski2015]
            alg_params["T0"] = custome_params["T0"] if custome_params is not None else 3000 
            
            alg_MC = MusicalChairs(alg_params)
            self.algorithms.append(alg_MC)
            
            # to record the learning results of alg_MC
            result_MC = ResultMultiPlayers(algo_type, 
                                           self.context_set, self.nbPlayers, self.nbArms, self.horizon)
            self.result_recorders.append(result_MC)
            
            self.alg_names.append(algo_type)
        
        elif algo_type == 'SIC-MMAB': #str(SICMMB)
            alg_params["horizon"] = self.horizon
            alg_SICMMAB = SICMMAB(alg_params)
            self.algorithms.append(alg_SICMMAB)
            
            # to record the learning results of alg_MC
            result_SICMMAB = ResultMultiPlayers(algo_type, 
                                           self.context_set, self.nbPlayers, self.nbArms, self.horizon)
            self.result_recorders.append(result_SICMMAB)
            
            self.alg_names.append(algo_type)
            
        elif algo_type == 'Hungarian': #str(Hungarian)
            alg_Hungarian = Hungarian(alg_params)
            self.algorithms.append(alg_Hungarian)
             
            result_hungarian = ResultMultiPlayers(algo_type, 
                                                  self.context_set, self.nbPlayers, self.nbArms, self.horizon)
            self.result_recorders.append(result_hungarian)
            
            self.alg_names.append(algo_type)
            
        elif algo_type == 'Static Hungarian':
            game_env = {}
            
            array_context, array_prob = self.mp_mab_env.get_discrete_context_prob()
            alg_params["array_context"] = array_context
            alg_params["array_prob"] = array_prob
            
            for context in self.context_set:
                 lower, upper, means, variance = self.mp_mab_env.get_param(context)                 
                 game_env[context] = means

            alg_params["mean_game_env"] = game_env
            
            alg_SHungarian = StaticHungarian(alg_params)
            self.algorithms.append(alg_SHungarian)
             
            result_static_hungarian = ResultMultiPlayers(algo_type, 
                                                         self.context_set, self.nbPlayers, self.nbArms, self.horizon)
            self.result_recorders.append(result_static_hungarian)
            
            self.alg_names.append(algo_type)
        elif  algo_type == 'Nonobservable-context Hungarian':
            # when the algorithm is not able to observe the context (side information)
            # the algorithm provides a optimal result in terms of normal MP-MAB            
            game_env = {}            
            game_mean = np.zeros((self.nbPlayers,self.nbArms))
            
            array_context, array_prob = self.mp_mab_env.get_discrete_context_prob()
            alg_params["array_context"] = array_context
            alg_params["array_prob"] = array_prob

            for context_id in range(len(array_context)):
                lower, upper, means, variance = self.mp_mab_env.get_param(array_context[context_id]) 
                game_mean = game_mean + means * array_prob[context_id]
            
            for context in self.context_set:
                 lower, upper, means, variance = self.mp_mab_env.get_param(context)                 
                 game_env[context] = game_mean

            alg_params["mean_game_env"] = game_env
            
            alg_SHungarian = StaticHungarian(alg_params)
            self.algorithms.append(alg_SHungarian)
             
            result_static_hungarian = ResultMultiPlayers(algo_type, 
                                                         self.context_set, self.nbPlayers, self.nbArms, self.horizon)
            self.result_recorders.append(result_static_hungarian)  
            
            self.alg_names.append(algo_type)
        elif algo_type == 'Game of Thrones':
            alg_params["horizon"] = self.horizon
            
            alg_params["c1"] = custome_params["c1"] if custome_params is not None else 100
            alg_params["c2"] = custome_params["c2"] if custome_params is not None else 5
            alg_params["c3"] = custome_params["c3"] if custome_params is not None else 1
            
            alg_params["epsilon"] = custome_params["epsilon"] if custome_params is not None else 0.1
            alg_params["delta"] = custome_params["delta"] if custome_params is not None else 2
            
            alg_GoT = GameofThrone(alg_params)
            self.algorithms.append(alg_GoT)
                        
            result_GoT = ResultMultiPlayers(algo_type, 
                                            self.context_set, self.nbPlayers, self.nbArms, self.horizon) 
            self.result_recorders.append(result_GoT)
            
            self.alg_names.append(algo_type)
        elif algo_type == "SOC":
            alg_params["delta"] = custome_params["delta"] if custome_params is not None else 0.1
            
            alg_SOC = SOC(alg_params)
            self.algorithms.append(alg_SOC)
            
            result_SOC = ResultMultiPlayers(algo_type, 
                                            self.context_set, self.nbPlayers, self.nbArms, self.horizon) 
            self.result_recorders.append(result_SOC)
            
            self.alg_names.append(algo_type) # use the full name of 'Stable Orthogonal Allocation'
        elif algo_type == "ESE":
            alg_params["delta_R"] = custome_params["delta_R"] if custome_params is not None else 0.1
            
            alg_ESE = ESE(alg_params)
            self.algorithms.append(alg_ESE)
            
            result_ESE = ResultMultiPlayers(algo_type, 
                                            self.context_set, self.nbPlayers, self.nbArms, self.horizon) 
            self.result_recorders.append(result_ESE)
            
            self.alg_names.append(algo_type) # use the full name of 'Stable Orthogonal Allocation'
            
        else:
             #TODO: add other algorithms here
             print("The algorithm type '{}' is not identified".format(algo_type))    
    
    def reset_algorithms(self, horizon = None):
        """
        reset the internal states/recorders of the algorithms
        """
        if horizon is not None:
            if self.flag_pre_prepare:
                if self.horizon < horizon:
                    raise Exception("horizon exceeds the maximum recorded values")
                else:
                    self.horizon = horizon
            else:
                self.horizon = horizon
        
        for index in range(len(self.algorithms)):
            self.algorithms[index].reset(horizon)
            self.result_recorders[index].reset_record(horizon)
            
        self.flag_simulation_done = False

    #----- play the bandit game with all the registered algorithms
    def play_game(self, algorithm_ids=None, horizon=None, flag_progress_bar=False):
        """
        play_game() produces a single round of simulation results in a sequentail way.
        It also works if there is no pre-prepared environment.
        """
        self.reset_algorithms()
        
        alg_list = []
        recorder_list = []
        if algorithm_ids is None:
            alg_list = self.algorithms
            recorder_list = self.result_recorders
        else:
            alg_list = [self.algorithms[index] for index in algorithm_ids]
            recorder_list = [self.result_recorders[index] for index in algorithm_ids]
            
        if horizon is None:
            horizon = self.horizon
        
        if flag_progress_bar:
            progress_range = tqdm(range(horizon))
        else:
            progress_range = range(horizon)       
        
        for t in progress_range:
            # sample arms
            if self.flag_pre_prepare == True:
                context, arm_values = self.mp_mab_env.draw_sample(t)
            else:
                context, arm_values = self.mp_mab_env.draw_sample()         
                
            # trim the arm_value array if needed                
            arm_values = arm_values[:self.nbPlayers, :self.nbArms]
#            print("shape of arm_values: {}".format(np.shape(arm_values)))
            
            for alg_index in range(len(alg_list)):
               pulls, total_reward, sampled_rewards = alg_list[alg_index].learn_policy(arm_values, context, t)
               arm_choices = alg_list[alg_index].pulls2choices(pulls)
               action_collisions = alg_list[alg_index].resolve_collision(pulls)
               recorder_list[alg_index].store(t, context, arm_choices, sampled_rewards, total_reward, pulls, action_collisions)
               
        self.flag_simulation_done = True
                    
    #----- play the bandit game with all the registered algorithms in a parallel manner
    def play_game_parallel(self, algorithm_ids=None, horizon=None, flag_progress_bar=False, step=100):
        """
        play_game_parallel() is restricted to work for the pre-prepared environment only.
        The extral time used for pickling the data is not negligible. 
        Multiprocessing doesn't improve much the efficiency if len(algorithm_ids) is less than 3 for small horizons.
        """        
        assert self.flag_pre_prepare == True, "the environment has to be prepared"
        self.reset_algorithms()
        
        # for parallel computing on a sngle machine
        max_nb_processes = max(mp.cpu_count()-2, 1)
        task_pool = mp.Pool(processes = max_nb_processes)     
        
        alg_list = []
        recorder_list = []
        if algorithm_ids is None:
            alg_list = self.algorithms
            recorder_list = self.result_recorders
        else:
            alg_list = [self.algorithms[index] for index in algorithm_ids]
            recorder_list = [self.result_recorders[index] for index in algorithm_ids]
            
        if horizon is None:
            horizon = self.horizon
            
        results = []
        
        if flag_progress_bar == False:
            for alg_index in range(len(alg_list)):
                 res = task_pool.apply_async(self.async_simulation_work, 
                                               args = (horizon, alg_index, self.mp_mab_env, 
                                                       alg_list[alg_index], recorder_list[alg_index]))                
                 results.append(res)
                      
            task_pool.close()
            task_pool.join()
        else:
            manager = mp.Manager()  
            queue = manager.Queue()
            for alg_index in range(len(alg_list)):
                 res = task_pool.apply_async(self.async_simulation_work, 
                                               args = (horizon, alg_index, self.mp_mab_env, 
                                                       alg_list[alg_index], recorder_list[alg_index], queue, step))                
                 results.append(res)
                 
            # add the monitoring process
            print("single-shot: number of iteration: {}".format(len(alg_list)*horizon))
            # add the monitoring process
            proc = mp.Process(target=self.porgress_monitor, 
                              args=(queue, len(alg_list), horizon))
            
            # start the processes
            proc.start()            
            task_pool.close()
            task_pool.join()               
            queue.put(None)
            proc.join()
               
        # each task do not exchange info. with each other
        self.flag_simulation_done = True
        
        for res in results:
            recorder = res.get()
            recorder_list[recorder[0]] = recorder[1]
        
#        print("AlgEvaluator finishes parallelization")
  
    @staticmethod
    def async_simulation_work(horizon, alg_index, env, alg, recorder, queue=None, step=100):
        """
        async_simu_work() is restricted to be called in play_game_parallel() only.
        To avoid passing the pool member, we make it a static method.
        """  
        # each task is identified by a tuple (alg_index, horizon)
        progress_range = range(horizon)
            
        for t in progress_range:
            context, arm_values = env.draw_sample(t)
            
            arm_values= arm_values[:env.nbPlayers, :env.nbArms]
#            print("shape of arm_values: {}".format(np.shape(arm_values)))
                
            pulls, total_reward, sampled_rewards = alg.learn_policy(arm_values, context, t)
            arm_choices = alg.pulls2choices(pulls)
            action_collisions = alg.resolve_collision(pulls)
            recorder.store(t, context, arm_choices, sampled_rewards, total_reward, pulls, action_collisions)     
            
            if queue is not None:
                if t % step == 0:
                    queue.put_nowait(step)                 
                
        return (alg_index, recorder)
    
    def play_repeated_game(self, horizon_list, algorithm_ids=None, 
                           simulation_rounds=1, flag_progress_bar=False):
        """
        Play the game repeatedly with different horizons in single-process mode. 
        It only works with the pre-prepared environment.
        The recorder accompanying each algorithm do not work here,
        since they store only the results from the last run.
        
        play_repeated_game() return a dictionary with the keys:
            {'algorithm_name', 'reward_series', 'collision_series', 'horizon'},
        where 'reward_series', 'horizon' and 'collision_series' are 2D arrays,
        with the rows aligned with elements in 'algorithm_name'
        """
        assert self.flag_pre_prepare == True, "the environment has to be prepared"
        self.reset_algorithms()

        alg_names = self.get_alg_names(algorithm_ids)        
        # reward_series records the reward data for each algorithm 
        # in a form (len(algorithm_ids), simulation_rounds*len(horizon_list))
        # other records are defined in the same form
        if algorithm_ids==None:
            algorithm_ids = list(range(len(self.algorithms)))
        
        reward_series = np.zeros((len(algorithm_ids), simulation_rounds*len(horizon_list)))
        collision_series = np.zeros((len(algorithm_ids), simulation_rounds*len(horizon_list)))
        switching_count_series = np.zeros((len(algorithm_ids), simulation_rounds*len(horizon_list)))
        horizon_series = np.zeros((len(algorithm_ids), simulation_rounds*len(horizon_list)))
        
        # convert types (convert ndarray to list)
        if isinstance(horizon_list, list) != True:
            horizon_list = np.ndarray.tolist(horizon_list)
        
#        print("number of algorithms: {}".format(len(algorithm_ids)))
        
        if flag_progress_bar:
            progress_range = tqdm(range(simulation_rounds))
        else:
            progress_range = range(simulation_rounds)    
        
        for simu_index in progress_range:
            if flag_progress_bar == False:
                print("Simulation round {} of total rounds {}...".format(simu_index+1, simulation_rounds))       
            
            for horizon in horizon_list:
                self.play_game(algorithm_ids, horizon=int(horizon), flag_progress_bar=False) # could set to None
                    
                # example: for 3 algorithms, len(tmp_total_payoff) == 3
                tmp_total_payoff = self.get_total_payoff(algorithm_ids, horizon=int(horizon))
                tmp_total_collision = self.get_total_collision(algorithm_ids, horizon=int(horizon))
                tmp_total_switching = self.get_total_switching_count(algorithm_ids, horizon=int(horizon))
                
                idx_horizon = horizon_list.index(horizon)

                id_plays = simu_index * len(horizon_list) + idx_horizon
                # record the reward obtained in this single round, 
                # the following is prepared for a dataframe format                
                for id_alg in range(len(algorithm_ids)):
                    horizon_series[id_alg][id_plays] = horizon
                    reward_series[id_alg][id_plays] = tmp_total_payoff[id_alg]
                    collision_series[id_alg][id_plays] = tmp_total_collision[id_alg]#
                    switching_count_series[id_alg][id_plays] = tmp_total_switching[id_alg]
                
        simulation_results = {}                
        simulation_results['reward_series'] = reward_series
        simulation_results['collision_series'] = collision_series          
        simulation_results['switching_count_series'] = switching_count_series
        simulation_results['horizon'] = horizon_series
        simulation_results['algorithm_name'] = alg_names
        
        return simulation_results
    
    #----- play the bandit game with (all) the registered algorithms in a parallel manner
    def play_repeated_game_parallel(self, horizon_list, algorithm_ids=None, 
                                    simulation_rounds=1, flag_progress_bar=False, step=1):
        """
        parallel version of repeated_game_play(). 
        play_repeated_game_parallel() only works with the pre-prepared environment.
        """
        assert self.flag_pre_prepare == True, "the environment has to be prepared"
        self.reset_algorithms()        

        alg_list = []
        recorder_list = []
        if algorithm_ids is None:
            alg_list = self.algorithms
            recorder_list = self.result_recorders
        else:
            alg_list = [self.algorithms[index] for index in algorithm_ids]
            recorder_list = [self.result_recorders[index] for index in algorithm_ids]
                
        # for parallel computing on a sngle machine
        max_nb_processes = max(mp.cpu_count()-2, 1)        
        task_pool = mp.Pool(processes = max_nb_processes)       
            
        # add works to the task pool
        results = []        
        if flag_progress_bar == True:
            manager = mp.Manager()  
            queue = manager.Queue()
            for alg_index in range(len(alg_list)):    
                res = task_pool.apply_async(self.async_repeated_work, 
                                            args = (self.mp_mab_env, alg_list[alg_index], 
                                                    alg_index, horizon_list, recorder_list[alg_index], 
                                                    simulation_rounds, queue, step)) 
                # append the results
                results.append(res)
                
            # add the monitoring process
            proc = mp.Process(target=self.porgress_monitor, 
                              args=(queue, len(alg_list), simulation_rounds))
            # start the processes
            proc.start()            
            task_pool.close()
            task_pool.join()               
            queue.put(None)
            proc.join()            
        else:
            for alg_index in range(len(alg_list)): 
                res = task_pool.apply_async(self.async_repeated_work, 
                                            args = (self.mp_mab_env, alg_list[alg_index], 
                                                    alg_index, horizon_list, recorder_list[alg_index], 
                                                    simulation_rounds)) 
                # append the results
                results.append(res)            
            # start the processes
            task_pool.close()
            task_pool.join()             
               
        # each task do not exchange info. with each other
        self.flag_simulation_done = True
                
        # reward_series records the reward data for each algorithm 
        # in a form (len(algorithm_ids), simulation_rounds*len(horizon_list))
        # all other records are defined in the same form
        reward_series = np.empty((0, simulation_rounds*len(horizon_list)))
        collision_series = np.empty((0, simulation_rounds*len(horizon_list)))
        switching_count_series = np.empty((0, simulation_rounds*len(horizon_list)))
        horizon_series = np.zeros((0, simulation_rounds*len(horizon_list)))
        alg_indicators = []
        
        # re-organize the results of each algorithm        
        for res in results:            
            alg_id, recorder, reward, collision, switching_count, horizons = res.get()                        
            # fill the recorded data with the last-round result
            self.result_recorders[alg_id] = recorder                       
            
            # add a new row
            reward_series = np.append(reward_series, [reward], axis=0)
            collision_series = np.append(collision_series, [collision], axis=0)
            switching_count_series = np.append(switching_count_series, [switching_count], axis=0)
            horizon_series = np.append(horizon_series, [horizons], axis=0)
            
            alg_indicators.append(alg_id)

        simulation_results = {}                
        simulation_results['reward_series'] = reward_series       
        simulation_results['collision_series'] = collision_series            
        simulation_results['switching_count_series'] = switching_count_series      
        
        simulation_results['horizon'] = horizon_series
        simulation_results['algorithm_name'] = [self.alg_names[index] for index in alg_indicators] 

#        print("len of collision_series:{}".format((collision_series.shape)))
#        print("len of reward_series:{}".format((reward_series.shape)))
#        print("len of switching_count_series:{}".format((switching_count_series.shape)))
#        print("len of horizon_series:{}".format((horizon_series.shape)))
        
        return simulation_results   
     
    @staticmethod
    def async_repeated_work(env, algrithm, alg_index, horizon_list, recorder, simulation_rounds=1, queue=None, step=1):
        """
        async_repeated_work() is should be only called by repeated_game_play_parallel().
        To avoid passing the pool member, we make it a static method.
 
        - a task is identified by a tuple (algrithm, horizon_list)        
        - 'reward_series' records the reward data for algorithm identified by 'alg_index'
          in an 1-D array of len(simulation_rounds)*len(horizon_list)
        - other records are defined in the same form      
        """ 
        reward_series = np.zeros(simulation_rounds*len(horizon_list))
        collision_series = np.zeros(simulation_rounds*len(horizon_list))     
        switching_count_series = np.zeros(simulation_rounds*len(horizon_list))
        horizon_series = np.zeros(simulation_rounds*len(horizon_list))
        
        #convert horizon type to list if it is an ndarray
        if isinstance(horizon_list, list) != True:
            horizon_list = np.ndarray.tolist(horizon_list)
        
        for simu_index in range(simulation_rounds):                            
            for horizon in horizon_list:
                idx_horizon = horizon_list.index(horizon)
                
                # reset the algorithm
                algrithm.reset()
                recorder.reset_record()
                
                # play the game
                progress_range = range(int(horizon))
                # initialize the switching count records
                tmp_total_switching = 0
                
                # store the choices according to the contexts that they are in
                tmp_switch_dic = {}
                tmp_context_count = {}
                for context in env.context_set:
                    tmp_switch_dic[context] = np.zeros([int(horizon), env.nbPlayers])
                    tmp_context_count[context] = 0
                
                for t in progress_range:
                    context, arm_values = env.draw_sample(t)
                    
                    arm_values = arm_values[:env.nbPlayers, :env.nbArms]
                                
                    # all in arrays
                    pulls, total_reward, sampled_rewards = algrithm.learn_policy(arm_values, context, t)
                    arm_choices = algrithm.pulls2choices(pulls)
                    action_collisions = algrithm.resolve_collision(pulls)                    
                    
                    #get collision in arrays
                    id_nonzero = np.where(action_collisions != 0)
                    action_collisions[id_nonzero] = action_collisions[id_nonzero] - 1

                    recorder.store(t, context, arm_choices, sampled_rewards, total_reward, pulls, action_collisions)  
                    
                    # store choices according to contexts                    
                    tmp_switch_dic[context][tmp_context_count[context],:] = arm_choices
                    tmp_context_count[context] =  tmp_context_count[context] + 1                    
                    
                for context in env.context_set:
                    # count the switching for each context
#                    print("Contex: {}, shape: {}".format(context,  tmp_switch_dic[context].shape))
                    
                    for tt in range(1, tmp_context_count[context]+1):                
                        tmp_switching_count = np.sum(tmp_switch_dic[context][tt,:] != tmp_switch_dic[context][tt-1, :])
                        tmp_total_switching += tmp_switching_count     
            
                # compute directly instead of calling get_total_payoff()
                tmp_total_payoff = np.sum(recorder.total_rewards[:int(horizon)])
                tmp_total_collision = np.sum(recorder.collisions[:int(horizon)])                            
                
                id_plays = simu_index * len(horizon_list) + idx_horizon
               
                reward_series[id_plays] = tmp_total_payoff
                collision_series[id_plays] = tmp_total_collision
                switching_count_series[id_plays] = tmp_total_switching
                horizon_series[id_plays] = horizon
            
            if queue is not None:
                if simu_index % step == 0:                    
                    queue.put_nowait(step)
                
        return (alg_index, recorder, reward_series, collision_series, switching_count_series, horizon_series)
    
    @staticmethod
    def porgress_monitor(queue, nbAlgorithm, nbRound):
        """
        porgress_monitor() is added by the monitor process for updating the simulation progress bar.
        nbRound represents the total number of repeatitions in case of a repeated simulation,
        or the number of horizon in case of a single-shot simulation
        """ 
        pbar = tqdm(total = nbAlgorithm*nbRound)
        for item in iter(queue.get, None):        
            pbar.update(item)
    
    #----- utility functions
    def get_total_payoff(self, algorithm_ids = None, horizon = None):
        assert self.flag_simulation_done == True, "no simulation record is available"  
        
        recorder_list = []
        if algorithm_ids is None:                
            recorder_list = self.result_recorders
        else:                
            recorder_list = [self.result_recorders[index] for index in algorithm_ids]   
            
        if horizon is None:
            horizon = self.horizon
        else:
            assert self.horizon >= horizon, "not enough data for the given value of horizon"
            
        array_total_payoff = np.zeros(len(recorder_list))
        for index in range(len(recorder_list)):                
            array_total_payoff[index] = np.sum(recorder_list[index].total_rewards[:horizon])
                
        return array_total_payoff
        
    def get_total_collision(self, algorithm_ids = None, horizon = None):
        assert self.flag_simulation_done == True, "no simulation record is available"        
        
        recorder_list = []
        if algorithm_ids is None:                
            recorder_list = self.result_recorders
        else:                
            recorder_list = [self.result_recorders[index] for index in algorithm_ids]   
        
        if horizon is None:
            horizon = self.horizon
        else:
            assert self.horizon >= horizon, "not enough data for the given value of horizon"
            
        array_total_collision = np.zeros(len(recorder_list))
        for index in range(len(recorder_list)):    
            idx_nonzero = np.where(recorder_list[index].collisions != 0)
            
            recorder_list[index].collisions[idx_nonzero] = recorder_list[index].collisions[idx_nonzero] - 1
            array_total_collision[index] = np.sum(recorder_list[index].collisions[:horizon])
            
        return array_total_collision
    
    def get_total_switching_count(self, algorithm_ids = None, horizon = None):
        """
        get the action switching count of the given list of algorithms,
        we do it w/r to the context 
        """
        assert self.flag_simulation_done == True, "no simulation record is available"
        
        recorder_list = []
        if algorithm_ids is None:                
            recorder_list = self.result_recorders
        else:                
            recorder_list = [self.result_recorders[index] for index in algorithm_ids]   
            
        if horizon is None:
            horizon = self.horizon
        else:
            assert self.horizon >= horizon, "not enough data for the given value of horizon"
            
        array_total_switching_count = np.zeros(len(recorder_list)) # with a number of the algorithms
        for index in range(len(recorder_list)):    
            total_switching_count = 0
            # we add choices into lists w/r to contexts
            tmp_switch_dic = {}
            tmp_context_count = {}
            for context in self.context_set:
                # we allocate a bit more than needed
                tmp_switch_dic[context] = np.zeros([horizon, self.nbPlayers])
                tmp_context_count[context] = 0
            
            # separate the action choices according to contexts
            for tt in range(0, horizon):
                context = self.result_recorders[index].context_history[tt]
                tmp_switch_dic[context][tmp_context_count[context],:] = self.result_recorders[index].choices[:,tt]
                tmp_context_count[context] = tmp_context_count[context] + 1
                
            for context in self.context_set:
                # count the switching for each context
                for tt in range(1, tmp_context_count[context]+1):                
                    tmp_switching_count = np.sum(tmp_switch_dic[context][tt,:] != tmp_switch_dic[context][tt-1,:])
                    total_switching_count += tmp_switching_count
                
            array_total_switching_count[index] = total_switching_count
            
        return array_total_switching_count
        
    def get_alg_names(self, algorithm_ids = None):
        """
        get the name list of the given algorithms
        """        
        if algorithm_ids is None:                
            name_list = self.alg_names
        else:                
            name_list = [self.alg_names[index] for index in algorithm_ids] 
            
        return name_list
    
    #----- plotting
    def plot_rewards(self, algorithm_ids = None, horizon = None, save_fig = False, save_data = False):
        if self.flag_simulation_done == False:
            print("No simulation results are ready")
        else:
            recorder_list = []
            if algorithm_ids is None:                
                recorder_list = self.result_recorders
            else:                
                recorder_list = [self.result_recorders[index] for index in algorithm_ids]
        
            recorder_list[0].plot_cumu_rewards(horizon, other_results=recorder_list[1:], save_fig=save_fig, save_data=save_data)
            recorder_list[0].plot_avg_reward(horizon, other_results=recorder_list[1:], save_fig=save_fig, save_data=save_data)