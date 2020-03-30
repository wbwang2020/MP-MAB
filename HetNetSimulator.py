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

# This file implements a simple heterogeneous network with underlying macro-cell UEs 
# working in a typical 5G cell, and the overlaying IoT devices working in a narrow-bandwidth (NB)
# mode. IoT devices are placed randomly at fixed locations and macro-cell UEs are moving
# randomly according to a Gauss—Markov model.

__author__ = "Wenbo Wang"

from MPMAB import MP_MAB

import scipy
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from plotutils import prepare_file_name

if __name__ == '__main__':
    print("Warning: this script 'HetNetSimulator.py' is NOT executable..")  # DEBUG
    exit(0)

class HomeBrewedHetNetEnv(MP_MAB):
    """
    The network simulator and its interface for MP_MAB.
    In the future version, we planned to incooperate existing simulators such as QuaDRiGa-5G for the channel models for the macro cell. 
    (see https://quadriga-channel-model.de/#Publications)
    For an example of a HetNet simulator over QuaDRiGa, see https://github.com/ICT-Embedded/5GNR-HetNet_Model
    Due to the time consumption of building it with matlab engine, we adopt a home-brewed HetNet simulator in this version.
    """    
    def __init__(self, context_set, nbArms=20, nbPlayers=10):
        """"""
        self.nbArms = nbArms # number of channels
        
        # for poisson point process it is the intensity of nodes for a square area of 1
        # for uniform distribution, it is the number of nodes
        assert nbPlayers<=nbArms, "the number of channels should be no less than the number of devices."
        self.nbPlayers = nbPlayers 
        self.nbArms = nbArms
        
        self.context_set = context_set# 
        self.prob_LoS = np.zeros(len(context_set))
        self.prob_context = np.zeros(len(context_set))
        
        self.current_arm_value = np.zeros((nbPlayers, nbArms))
        self.current_context = None
        
        self.arms = {};
        self.horizon = 0
        
        self.flag_mmWave = True
        
        self.ue_position = []
        self.bs_position = []
        
        self.nb_mue = 0
        self.mue_position = []# macro cell UE
        self.mue_mean_vel = []
        self.mue_mean_dir = []
        
        #basic parameters of channel, not exposed to the parameter setting yet
        self.frequence = 28e9 # 28GHz 
        self.nb_UPBC = 4# number of unique pointing beans combined
        self.wf_A = 0.07# weighting factor via MMSE for fitting BC-CI path loss model
        self.ue_shadow_variance = np.zeros((nbPlayers, nbArms)) # currently based on an arbitrary value, e.g., 9
        self.ue_fading_variance = np.ones((nbPlayers, nbArms)) # currently based on an arbitrary value
        self.mobile_alpha = 0.3
        
        self.mue_shadow_variance = np.zeros((len(context_set))) # the same across arms
        
        """
        The path loss exponent model is silightly different w.r.t. to different experiments in the literature.
        According to "Path Loss, Shadow Fading, and Line-Of-Sight Probability Models for 5G Urban Macro-Cellular Scenarios", [Sun2015],
        PLE_LoS = 2.1 and PLE_NLoS = 2.6 for the CI model in the 28GHz-urban macro-cellular scenario
        """        
        self.PLE_LoS = 2 # path loss exponent LoS
        self.PLE_NLoS = 3 # path loss exponent NLoS
        self.mue_power = 10 * np.random.uniform(low=0.5, high=1.0, size=len(context_set)) # 40 dBm, 10w
        self.ue_power = 1 # 30 dBm, 1w
        self.atenna_gain = 3 #dBi        
        self.noise = 5e-17 # Watt 
        
        # for beamforming, the oversampling factor is 1
        # we consider the beamforming vector to be randomly choosen, 
        # this project does not aim to provide mechanisms of optimizing it
        self.F = np.zeros([self.nb_UPBC, self.nb_UPBC], dtype=complex)
        theta = np.pi * np.arange(start=0., stop=1., step=1./self.nb_UPBC) 
        # Beamforming codebook F
        for nn in np.arange(self.nb_UPBC):
            exponent = 1j * np.pi * np.cos(theta[nn]) * np.arange(self.nb_UPBC)            
            bf_vec = 1. / np.sqrt(self.nb_UPBC) * np.exp(exponent)                        
            self.F[:,nn] = bf_vec[nn]
        
        self.mue_cb_idx = np.random.randint(self.nb_UPBC)
        # to simplify the process of computation, we consider the IoT devices are using the same ones
        # it does not affect the simulation results
        self.iot_cb_idx = np.random.randint(self.nb_UPBC) 
        
        # recorder of the pre-sampled arm values
        self.arm_values = {}
        self.max_arm_value ={} # recording the maximum rate for normalization for each context along the time horizon
        for context in self.context_set:
            self.arm_values[context] = []
            self.max_arm_value[context] = []
        
        self.flag_sample_prepared = False
        
    @classmethod
    def HetNet_mab(cls, context_set, nbArms, nbPlayers, hetnet_params):
        """
        A number of parameters are hardcoded for the purpose of simplification. 
        However, they can be easily exposed to the upper layer by moving into 'hetnet_params'
        
        """
        hetnet_inst = cls(context_set, nbArms, nbPlayers)
        
        hetnet_inst.horizon = hetnet_params['horizon']        
        hetnet_inst.flag_mmWave = hetnet_params['enabel mmWave']
        
        cell_range = hetnet_params['cell range'] if 'cell range' in hetnet_params.keys() else 200
        hetnet_inst.bs_position = np.array([0.5 * cell_range, 0.5 * cell_range]) # always placed at the center     
        hetnet_inst.ue_position, new_nbPlayer = hetnet_inst.initialize_UE_position(nbPlayers=nbPlayers, distance = cell_range,
                                  dist_model=hetnet_params['dist_model'] if 'dist_model' in hetnet_params.keys() else 0)
        
        hetnet_inst.mue_position, new_nbMUE = hetnet_inst.initialize_UE_position(nbPlayers=len(hetnet_inst.context_set), 
                                                                      distance=cell_range, dist_model=0)
        
        # randomly set shadowing variances of ue's, as an array of (nbUE-nbChannel)
        shadow_vr_base = 2.0 if 'shadow_vr' not in hetnet_params.keys() else hetnet_params['shadow_vr']        
        hetnet_inst.ue_shadow_variance = np.random.uniform(size=(nbPlayers, nbArms))*shadow_vr_base
        hetnet_inst.mue_shadow_variance = np.random.uniform(size=len(context_set))*shadow_vr_base

        fading_vr_base = 1.0 if 'fading_vr' not in hetnet_params.keys() else hetnet_params['fading_vr'] 
        hetnet_inst.ue_fading_variance =  np.random.uniform(size=(nbPlayers, nbArms))*fading_vr_base        
        
        # assume that different context has different probability of LoS path
        hetnet_inst.set_discrete_context_prob(hetnet_params['context_prob'], hetnet_params['los_prob']) 
                                  
        nb_MUE = len(hetnet_inst.prob_context)
        hetnet_inst.mue_mean_vel, hetnet_inst.mue_mean_dir = hetnet_inst.initialize_UE_mobile_model(nb_MUE, scale_velocity=0.1)
        
        hetnet_inst.mue_vel = np.zeros(nb_MUE)
        hetnet_inst.mue_dir = np.zeros(nb_MUE)
        
        hetnet_inst.vel_base = 1.0 if 'vel_base' not in hetnet_params.keys() else hetnet_params['vel_base']        
        
        return hetnet_inst
    
    def set_discrete_context_prob(self, context_prob, los_prob):
        """
        assign arbitrary probabilities to contexts
        """
        if set(context_prob.keys()) != self.context_set:
            raise Exception("probability values do not match the set of context")
        
        self.context_array = np.array(list(context_prob.keys()))
        
        # probability of different MUE/UE in neighbor cells transmitting
        self.prob_context = np.array(list(context_prob.values()))
        self.prob_context = self.prob_context / np.sum(self.prob_context) # normalize 
        
        # probability of different MUE to the receiving AP
        # this is to simulate the situation that transmissions from different MUE occupy the channels in the cell
        self.prob_LoS = np.array(list(los_prob.values()))
        self.prob_LoS = self.prob_LoS / np.sum(self.prob_LoS) # normalize          


    def initialize_UE_position(self, nbPlayers, distance=200, dist_model=0):
        """
        initialize the positions of IoT devices and UEs
        """
        if dist_model == 1:# PPP distribution        
            #TODO: the input number of nodes may not be equal to N according to the PPP distribution
            # we need to update the player number self.nbPlayers
            # do not call this branch in this version
            N = scipy.stats.poisson( nbPlayers*1 ).rvs()            

        else: # uniform distribution, TODO: add new distribution model here
            N = nbPlayers
            
        x = scipy.stats.uniform.rvs(0, 1,((N,1)))*distance
        y = scipy.stats.uniform.rvs(0, 1,((N,1)))*distance       

        ue_position = np.hstack((x,y)).T
        
        return ue_position, N
    
    def initialize_UE_mobile_model(self, nbPlayers, scale_velocity=1):
        ue_mean_vel = np.random.uniform(nbPlayers)*scale_velocity
        ue_direction = np.random.uniform(nbPlayers)*np.pi*2
        
        return ue_mean_vel, ue_direction
            
    """Draw samples"""
    def draw_sample(self, t=None):        
        """
        draw a new sample        
        """
        context_id_array = np.arange(start=0, stop=len(self.context_array))             
        id_context = np.random.choice(a=context_id_array, size=None, p=self.prob_context) # choose the ID of MUE
        self.current_context = self.context_array[id_context] # get the context value 
        
        if t == None:            
            # update all MUEs' positions
            self.mue_position, self.mue_vel, self.mue_dir = self.update_ue_position(self.mue_position, self.mue_vel, 
                                                                self.mue_dir, self.mobile_alpha, self.mue_mean_vel, self.mue_mean_dir)
                        
            current_arm_value = self.compute_device_rate(id_context)
            
            # normalization
            self.current_arm_value = current_arm_value / np.max(current_arm_value)
        else:
            if self.flag_sample_prepared == False:
                raise Exception("samples are not prepared")
            else:
                # draw samples from the stored data
                self.current_arm_value = self.arm_values[self.current_context][t]                
        
        return self.current_context, self.current_arm_value # we only return part of the real data
         

    def prepare_samples(self, horizon, flag_progress_bar=True):
        """
        Prepare the samples along the time horizon in advance.
        The sequential generation of UE positions would be the most significant bottleneck 
        for the simulation. 
        """
        if horizon <= 0:
            raise Exception("Input horizon is not valid")
                    
        self.horizon = horizon
        
        if flag_progress_bar:
            progress_range = tqdm(range(horizon))
        else:
            progress_range = range(horizon)
        
        for time in progress_range:
            # update position first
            self.mue_position, self.mue_vel, self.mue_dir = self.update_ue_position(self.mue_position, self.mue_vel, 
                                                            self.mue_dir, self.mobile_alpha, self.mue_mean_vel, self.mue_mean_dir)
            # the positions are the same w.r.t. each channel, but the shadowing/fading parameters are different
            for context in self.context_set:
                id_context = self.context_array.tolist().index(context) #np.where(self.context_array == context)
                
                rates = self.compute_device_rate(id_context)
                
                # normalization
                current_max_rate = np.max(rates)
                normalized_rate = rates / current_max_rate
                # record the normalized rate matrix at "time"
                self.arm_values[context].append(normalized_rate)
                self.max_arm_value[context].append(current_max_rate) #added @ 2020.02.21
                    
        self.flag_sample_prepared = True
        
    """
    methods used in draw_sample()
    """
    def update_ue_position(self, ue_position, ue_vel, ue_dir, mobil_alpha, ue_mean_vel, ue_mean_dir):         
         # Gauss—Markov mobility model, Chapter 2.5. Gauss—Markov "A survey of mobility models for ad hoc network research", [Camp2002]         
         # Calculate the new velocity and direction values using the Gauss-Markov formula:
         # new_val = alpha*old_val + (1-alpha)*mean_val + sqrt(1-alpha^2)*rv
         # where rv is a random number sampled from a normal (gaussian) distribution
         # reference code (ns-3): https://www.nsnam.org/docs/doxygen/gauss-markov-mobility-model_8cc_source.html
         one_minus_alpha = 1 - mobil_alpha
         sqrt_alpha = np.sqrt(1 - mobil_alpha**2)
         
         rv = np.random.normal(size=len(ue_vel)) * self.vel_base # velocity
         rd = np.random.normal(size=len(ue_vel)) # angle
         
         # random value, default parameters: mean = 0, and variance = 1
         ue_vel = mobil_alpha * ue_vel + one_minus_alpha * ue_mean_vel + sqrt_alpha * rv
         ue_dir = mobil_alpha * ue_dir + one_minus_alpha * ue_mean_dir + sqrt_alpha * rd
         
         cos_dir = np.cos(ue_dir)
         sin_dir = np.sin(ue_dir)
         
         x = ue_position[0,:] + ue_vel * cos_dir
         y = ue_position[1,:] + ue_vel * sin_dir
         
         ue_position = np.vstack((x,y))
         
         return ue_position, ue_vel, ue_dir
         
    # used for sampling channels gains
    def update_pathloss_db(self, ue_pos, bs_pos, flag_LoS=False):    
        #update the pathloss of the IoT devices and the macrocell UE
        if self.flag_mmWave == True:                
            if flag_LoS == True:
                pl_db = self.path_loss_dB_mmWave(ue_pos, bs_pos, self.PLE_LoS)
            else:
                pl_db = self.path_loss_dB_mmWave(ue_pos, bs_pos, self.PLE_NLoS)
            
#            pl = 10 ** (pl_db / 10.)
        else:
            pl_db = self.path_loss_dB_cost231(ue_pos, bs_pos)
#            pl = 10 ** (pl_db / 10.)
            
        return pl_db # path loss in dB
            
    # we may need to compute different ue/device-BS pairs
    def path_loss_dB_mmWave(self, ue_position, bs_position, PLE):
        """
        Based on IEEE TWC paper "Directional Radio Propagation Path Loss Models for Millimeter-Wave 
        Wireless Networks in the 28-, 60-, and 73-GHz Bands", Oct. 2016 [Sulyman2016]
        Nr is the number of unique pointing beams combined, Nr = 3,4,5
        """
        #PLE = 2 for LoS, 4 for NLoS, see self.PLE_LoS, self.PLE_NLoS        
        c = 3e8 # light speed
        
        # to align the notations with the equations in the refernece [Sulyman2016]
        A = self.wf_A 
        nr = self.nb_UPBC
        fc = self.frequence # in Hz
        
        if ue_position.ndim == 1:
            pass # single ue, don't have to do anything
        else:            
            bs_position = np.broadcast_to(bs_position, (ue_position.shape[::-1])).T
        
        dist = np.linalg.norm(ue_position-bs_position, axis=0) # along the rows          
        
#         fspl = 32.4 + 20 * np.log10(fc / 1e9) # fc in GHz, Eq (1a) of 2016 [Sulyman2016], equivalent equation
        fspl = 20 * np.log10((4*np.pi*dist*fc) / c) # Eq (1a) of 2016 [Sulyman2016]
        pl = fspl + 10 * PLE * np.log10(dist) * (1 - A*np.log2(nr)) # Eq (8) of 2016 [Sulyman2016]    
    
        return pl # in dB    
    
    def path_loss_dB_cost231(self, ue_position, bs_position):
        """
        reference: A.2.2 COST 231 Extension to Hata Model, Channel Models A Tutorial, [Jain2007]
        code reference: https://www.mathworks.com/matlabcentral/fileexchange/21795-hata-and-cost231-hata-models
        """
        fc = self.frequence

        dist =np.linalg.norm(ue_position-bs_position, axis=1)    

        h_BS = 20 #  effective base station antenna height
        h_UE = 1.5 # mobile station antenna height
     
        # COST231        
        C = 3
        
        # equation: ahMS = (1.1 * log10(f) - 0.7) * hMS - (1.56 * log10(f) - 0.8);
        ahMS = (1.1 * np.log10(fc/1e6) - 0.7)*h_UE - (1.56*np.log10(fc/1e6) - 0.8)
        
        # equation:  L50dB = 46.3 + 33.9 * log10(f) - 13.82 * log10(hBSef) - ahMS + (44.9 - 6.55 * log10(hBSef)) * log(d) + C;
        # f is in MHz, dist is in km        
        pl = 46.3 + 33.9 * np.log10(fc/1e6) + 13.82 * np.log10(h_BS) - ahMS + (44.9 - 6.55 * np.log10(h_BS)) * np.log10(dist/1000.) + C
                
        return pl # in dB
        
    def update_shadow(self, shadow_mean, shadow_var, ue_number):        
        """
        log-normal shadowing
        """
        # ue_number is used in case the shadowing parameters are the same
        chi_shadow = np.random.normal(loc=shadow_mean, scale=shadow_var, size=ue_number) # log-normal shadowing in dB
        
        return chi_shadow # in dB
        

    def update_fast_fading(self, ue_number, rb_number, fading_variance, fading_type=0):
        """
        Rayleigh fading
        """
        if fading_type == 0:
            """
            Rayleigh fading,     
            """
            if rb_number > 1:
                scale = np.broadcast_to(fading_variance, (ue_number, rb_number))
                hf = 1/np.sqrt(2*scale) * (np.random.normal(scale = scale, size = (ue_number, rb_number)) 
                              + 1j* np.random.normal(scale = scale, size = (ue_number, rb_number)))
            else:
                scale = fading_variance
                hf = 1/np.sqrt(2*scale) * (np.random.normal(scale=scale, size=ue_number) 
                              + 1j* np.random.normal(scale=scale, size=ue_number))
            
            h_fading = 20 * np.log10(np.abs(hf)) # in dB
        else:
            #implement other fast fading model here
            raise NotImplementedError("fast fading types not supported")
        
        return h_fading # in dB
        

    def update_MUE_channels(self, mue_position, mue_shadow_variance, flag_LoS=False):
        """
        update_MUE_channels() and update_IoT_channels() are functions called by compute_device_rate()
        """
        # update_MUE_channels() is supposed to update a single MUE's (according to the context id) channel information
        # multiple MUE is also supported
        #
        pl = self.update_pathloss_db(mue_position, self.bs_position, flag_LoS)
        sh = self.update_shadow(shadow_mean=0, shadow_var=mue_shadow_variance, ue_number=1)
        ff = 0# compared with the path loss, we ignore fast fading here
                
        if mue_position.ndim == 1:
            # to check if we compute for a single MUE or multiple ones
            nb_mue = 1
        else:
            nb_mue = mue_position.shape[0]
            
        
        channel_gains = np.array((nb_mue, self.nb_UPBC), dtype=complex)
        
        if nb_mue == 1:
            channel_gains = self.update_channel_gain(pl, sh, ff, self.atenna_gain, flag_LoS)
        else:            
            for ii in range(nb_mue):
                channel_gains[ii,:] = self.update_channel_gain(pl[ii], sh[ii], ff[ii], self.atenna_gain, flag_LoS)
               
        return channel_gains 
    
    def update_IoT_channels(self, flag_LoS=False):
        # we assume that the iot devices do not move
        pl = self.update_pathloss_db(self.ue_position, self.bs_position, flag_LoS) # the same for each player
        
        channel_gains = np.zeros((self.nbPlayers, self.nbArms, self.nb_UPBC), dtype=complex)
        for id_arm in range(self.nbArms):
            # not the same for each channel/arm
            sh = self.update_shadow(shadow_mean=0, shadow_var=self.ue_shadow_variance[:,id_arm], ue_number=self.nbPlayers) 
            ff = self.update_fast_fading(self.nbPlayers, 1, self.ue_fading_variance[:,id_arm]) # not the same for each channel/arm
            
            for ii in range(self.nbPlayers):            
                channel_gains[ii, id_arm, :] = self.update_channel_gain(pl[ii], sh[ii], ff[ii], self.atenna_gain, flag_LoS) #pl + sh - ff
        
        return channel_gains    
    
    def update_channel_gain(self, pl, sh, ff, atenna_gain, flag_LoS): 
        """
        consider a uniform linear array (ULA) with nb_UPBC antennas, 
        the steering vector of the array towards direction θ is denoted as theta
        """        
        path_loss = 10 ** (pl / 10.)

        vb = np.zeros(self.nb_UPBC, dtype=complex)        
        # v is the array vector                    
        if (flag_LoS == True):
            Np = 1
            vb[0] = 1. / np.sqrt(path_loss)
        else:
            # 
            Np = self.nb_UPBC
            vb = (np.random.normal(size=Np) + 1j * np.random.normal(size=Np)) / np.sqrt(path_loss)

        # randomly generated
        theta = np.random.uniform(low=0, high=np.pi, size=Np)                
        rho = 10 ** ((atenna_gain + sh + ff ) / 10.)
        
        # initialize the channel as a complex variable.
        h_ch = np.zeros(self.nb_UPBC, dtype=complex)
        
        for path in np.arange(Np):
            exponent = 1j * np.pi * np.cos(theta[path]) * np.arange(self.nb_UPBC)
            
            bf_vec = 1. / np.sqrt(self.nb_UPBC) * np.exp(exponent)
            h_ch = h_ch + bf_vec[path] / rho * bf_vec.T # scalar multiplication into a vector
        
        h_ch = h_ch * np.sqrt(self.nb_UPBC)
        
        return h_ch
    
    def compute_device_rate(self, id_context):
        id_LoS = np.random.choice([0,1], p=[self.prob_LoS[id_context], 1-self.prob_LoS[id_context]])
         
        mue_channel_gain = self.update_MUE_channels(self.mue_position[:, id_context], self.mue_shadow_variance[id_context],
                                                     flag_LoS=(id_LoS == 0)) # part of the context, interference
         
        iot_channel_gains = self.update_IoT_channels(flag_LoS=False)
         
        # get the channel capacity w.r.t. each IoT devices over each arm/channel
        interference_power = self.mue_power[id_context] * abs(np.dot(mue_channel_gain.conj(), self.F[:, self.mue_cb_idx])) ** 2
        
        iot_received_power = np.zeros((self.nbPlayers, self.nbArms))# 2D matrix, columns correspond to each channel
        for player_id in range(self.nbPlayers):
            for ch_id in range(self.nbArms):
                iot_received_power[player_id][ch_id] = self.ue_power * abs(np.dot(iot_channel_gains[player_id, ch_id,:].conj(), 
                              self.F[:, self.iot_cb_idx])) ** 2   

        mue_ipn = interference_power + self.noise #interference plus noise, scalar

        # should be a (nbPlayer, nbArm) matrix
        rates = np.log2(1 + np.divide(iot_received_power, mue_ipn))
         
        # update the rate value for all players over all arms         
        return rates    

    """utility functions"""   
    # helper_plot_ue_posiiton() is used only for debugging
    def helper_plot_ue_posiiton(self):
        """
        For debugging purpose
        """
        plt.figure(figsize=(4,3))
        plt.scatter(self.ue_position[0,:], self.ue_position[1,:], edgecolor='b', facecolor='none', alpha=0.5 )     
        plt.scatter(self.mue_position[0,:], self.mue_position[1,:], edgecolor='r', facecolor='none', alpha=0.5 ) 
     
    def get_discrete_context_prob(self):        
        return self.prob_context
            
    def get_param(self, context):
        # it is difficult to get the rate statisitics of the UEs over each channel
        raise NotImplementedError("get_param() is not campatible with class HomeBrewedHetNetEnv.") 
        
    def get_current_param(self, t=None):
        """ 
        Get the current sampling parameters of arms in the given context.
        """
        raise NotImplementedError("This method get_current_param() is not campatible with class HomeBrewedHetNetEnv.") 
        
    def save_environment(self, file_name=None):
        #TODO: not fully tested yet, not used
        if self.flag_sample_prepared == False:
            print("No data is prepared")
        else:       
            # we cannot select the path yet, put the file to the default directory "\results" of the current path            
            file_path = prepare_file_name("{}-{}".format(file_name if file_name is not None else "", "env"), 
                                          alg_name = None, ext_format = "mat")
                        
            scipy.io.savemat(file_path, self.arm_values)
        
    def load_environment(self, file_path, horizon=None):
        #TODO: not fully tested yet, not used
        try:            
            self.arm_values = scipy.io.loadmat(file_path)
        except:
            print("No data is loaded")
            
        self.flag_sample_prepared = True