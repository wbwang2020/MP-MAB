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

This file defines the data recorder and parts of the virtualization mechanisms in our simulations.
class ResultMultiPlayers
"""

# This file implements the data recorder for each single MAB algorithm

__author__ = "Wenbo Wang"

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from plotutils import make_markers, make_palette, display_legend, prepare_file_name
from datetime import datetime

if __name__ == '__main__':
    print("Warning: this script 'PlayerResult.py' is NOT executable..")  # DEBUG
    exit(0)
    
DELTA_T_PLOT = 50
FIGURE_SIZE = (5, 3.75)

class ResultMultiPlayers(object):
    """ ResultMultiPlayers accumulators, for the multi-players case. """
    
    def __init__(self, alg_name, context_set, player_no, arm_no, horizon):
        """ Create ResultMultiPlayers."""
        self.alg_name = alg_name
        
        self.nbPlayer = player_no
        self.nbArm = arm_no
        self.context_set = context_set
        self.horizon = horizon
        
        self.choices = np.zeros((player_no, horizon), dtype=int)  #: Store all the arm choices of all the players
        self.sampled_rewards = np.zeros((player_no, horizon))  #: Store all the rewards of all the players, to compute the mean
        self.total_rewards = np.zeros(horizon) 
        
        self.context_history = [None]*horizon
        
        self.pull_history = np.zeros((player_no, arm_no, horizon), dtype=int)  #: Is a map of 0-1 for players and arms
        self.collisions = np.zeros((arm_no, horizon), dtype=int)  #: Store the number of collisions on all the arms
        
        self.delta_t_plot = 1 if self.horizon <= 10000 else DELTA_T_PLOT

    def store(self, time, context, choices, sampled_rewards, total_rewards, pulls, collision=None):
        """ Store results."""
        self.context_history[time] = context
        
        self.choices[:, time] = choices
        self.sampled_rewards[:, time] = sampled_rewards
        self.total_rewards[time] = total_rewards

        self.pull_history[:, :, time] = pulls
        
        if collision is None:
            self.collisions[:, time] = 0
        else:
            self.collisions[:, time] = collision
            
    def reset_record(self, horizon=None):
        if horizon is not None:
            self.horizon = horizon
        
        self.choices = np.zeros((self.nbPlayer, self.horizon), dtype=int)  #: Store all the arm choices of all the players
        self.sampled_rewards = np.zeros((self.nbPlayer, self.horizon))  #: Store all the rewards of all the players, to compute the mean
        self.total_rewards = np.zeros(self.horizon) 
        
        self.context_history = [None]*self.horizon
        
        self.pull_history = np.zeros((self.nbPlayer, self.nbArm, self.horizon), dtype=int)  #: Is a map of 0-1 for players and arms
        self.collisions = np.zeros((self.nbArm, self.horizon), dtype=int)  #: Store the number of collisions on all the arms
        
        
    def dump2disk(self, file_name=None):
        """Save the result into a Matlab .mat file"""       
        file_path = prepare_file_name(file_name, self.alg_name, "mat")
        
        scipy.io.savemat(file_path, mdict={"nbPlayer": self.nbPlayer, "nbArm": self.nbArm, "context_set": list(self.context_set),
                                           "horizon": self.horizon, "context_history": self.context_history, 
                                           "sampled_reward": self.sampled_rewards,
                                           "choices": self.choices, "collisions": self.collisions})
        
        
    """
    The following methods are used for plotting/saving figures.  
    Other figure plotting methods can be found in plotutils.py
    """
    def plot_cumu_rewards(self, horizon=None, other_results=None, semilogx=False, save_fig=False, save_data=False):
        #other_results are used for comparison with other algorithms
        if other_results is not None:
            #the other results should have the same player/arm numbers
            for idx in range(len(other_results)):
                nbPlayer = other_results[idx].nbPlayer
                nbArm = other_results[idx].nbArm
                
                if nbPlayer != self.nbPlayer or nbArm != self.nbArm:
                    raise Exception("environment does not match!")
                    
            nbCurves = self.nbPlayer * (1 + len(other_results))
        else:
            nbCurves = self.nbPlayer
            
        """Plot the decentralized rewards, for each player."""
        fig = plt.figure(figsize=FIGURE_SIZE)
        ymin = 0
        colors = make_palette(nbCurves)
        markers = make_markers(nbCurves)
        
        if horizon is None:
            horizon = self.horizon
        
        X = np.arange(start=0, stop=horizon, step=1)
        
        #plot the locally stored values
        cumu_rewards = np.cumsum(self.sampled_rewards, axis=1)

        curve_idx = 0
        for playerId in range(self.nbPlayer):            
            label = '{}: Player {:>2}'.format(self.alg_name, playerId + 1)
            Y = cumu_rewards[playerId, :horizon]
            Y = Y / (X+1)

            ymin = min(ymin, np.min(Y))
            if semilogx:
                plt.semilogx(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[curve_idx], 
                             marker=markers[curve_idx], markersize=5, markevery=(curve_idx / 50., 0.1), lw=1)
            else:
                plt.plot(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[curve_idx], 
                         marker=markers[curve_idx], markersize=5, markevery=(curve_idx / 50., 0.1), lw=1)
                
            curve_idx = curve_idx + 1
            
        if other_results is not None:
             for idx in range(len(other_results)):
                 cumu_rewards = np.cumsum(other_results[idx].sampled_rewards, axis=1)
                 for playerId in range(other_results[idx].nbPlayer):
                     label = '{}: Player {:>2}'.format(other_results[idx].alg_name, playerId + 1)
                     Y = cumu_rewards[playerId, :horizon]
                     Y = Y / (X+1)
                     ymin = min(ymin, np.min(Y))
                     if semilogx:
                         plt.semilogx(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[curve_idx], 
                                  marker=markers[curve_idx], markersize=5, markevery=(curve_idx / 50., 0.1), lw=1)
                     else:
                         plt.plot(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[curve_idx], 
                                  marker=markers[curve_idx], markersize=5, markevery=(curve_idx / 50., 0.1), lw=1)
                     
                     curve_idx = curve_idx + 1
                
        display_legend()
        plt.xlabel("Number of rounds", fontsize=10)
        plt.ylabel("Average reward over time", fontsize=10)

#        plt.title("Individual Average Rewards Over Time", fontsize=10)
        if save_data:
            print("saving figure...")
            self.dump2disk()
        
        if save_fig:
            self.save_figure(file_name = "indv_avg_result", fig=fig)
            
        return fig

    def plot_avg_reward(self, horizon=None, other_results=None, semilogx=False, save_fig=False, save_data=False):
         #other_results are used for comparison with other algorithms
        if other_results is not None:
            #the other results should have the same player/arm numbers                    
            nbCurves = 1 + len(other_results)
        else:
            nbCurves = 1
            
        """Plot the average rewards, for each player in each algorithm."""
        fig = plt.figure(figsize=FIGURE_SIZE)
        ymin = 0
        colors = make_palette(nbCurves)
        markers = make_markers(nbCurves)
        
        if horizon is None:
            horizon = self.horizon
            
        X = np.arange(start=0, stop=horizon, step=1)   
        
        #plot the locally stored values
        curve_idx = 0 
        cumu_rewards = np.cumsum(self.total_rewards[:horizon])

        label = '{}'.format(self.alg_name)
        Y = cumu_rewards / (X+1) / self.nbPlayer

        ymin = min(ymin, np.min(Y))
        if semilogx:
            plt.semilogx(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[curve_idx], 
                         marker=markers[curve_idx], markersize=5, markevery=(curve_idx / 50., 0.1), lw=1)
        else:
            plt.plot(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[curve_idx], 
                         marker=markers[curve_idx], markersize=5, markevery=(curve_idx / 50., 0.1), lw=1)
                
        if other_results is not None:            
            for idx in range(len(other_results)):
                curve_idx = curve_idx + 1
                cumu_rewards = np.cumsum(other_results[idx].total_rewards[:horizon])
                
                label = '{}'.format(other_results[idx].alg_name)
                Y = cumu_rewards / (X+1) / other_results[idx].nbPlayer
                
                ymin = min(ymin, np.min(Y))
                if semilogx:
                    plt.semilogx(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[curve_idx], 
                                 marker=markers[curve_idx], markersize=5, markevery=(curve_idx / 50., 0.1), lw=1)
                else:
                    plt.plot(X[::self.delta_t_plot], Y[::self.delta_t_plot], label=label, color=colors[curve_idx], 
                                 marker=markers[curve_idx], markersize=5, markevery=(curve_idx / 50., 0.1), lw=1)               
                
        display_legend()
        plt.xlabel("Number of rounds", fontsize=10)
        plt.ylabel("Average reward over time", fontsize=10)
#        plt.title("Individual Average Rewards Over Time", fontsize=10)

        if save_data:
            print("saving figure data...")
            self.dump2disk()
        
        if save_fig:
            print("saving figure...")
            self.save_figure(file_name = "avg_result", fig=fig)

        return fig        
                    
    def save_figure(self, file_name=None, formats={'pdf', 'png'}, fig=None):
        now = datetime.now()
        
        for form in formats:               
            path = prepare_file_name(file_name, self.alg_name, form)
            try:
                current_time = now.strftime("%H:%M:%S")
                plt.savefig(path, bbox_inches="tight")
                print("Figure saved! {} at {} ...".format(path, current_time))   
            
            except Exception as exc:
                print("Could not save figure to {} due to error {}!".format(path, exc))  # DEBUG  
                