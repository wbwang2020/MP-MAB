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

# This file defines the plotting methods for the simulation.
# The configuration for pallette creation are partially inspired by the ones in the SMPyBandits project, 
# see plotsettings.py in SMPyBandits (https://github.com/SMPyBandits/SMPyBandits)


__author__ = "Wenbo Wang"

from datetime import datetime

import matplotlib as mpl
#from matplotlib.ticker import FuncFormatter 

import os, errno

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

#import numpy as np
import seaborn as sns
import pandas as pd

import numpy as np

#from pickle import dump as pickle_dump # alternative choice of dumping files

DPI = 120  #: DPI to use for the figures
FIGSIZE = (4,3) #: Figure size, in inches
#FIGSIZE = (5,4) #: Figure size, in inches

# Customize the colormap
HLS = True  #: Use the HLS mapping, or HUSL mapping
VIRIDIS = False  #: Use the Viridis colormap

# Bbox in inches. Only the given portion of the figure is saved. If 'tight', try to figure out the tight bbox of the figure.
BBOX_INCHES = "tight"  #: Use this parameter for bbox
BBOX_INCHES = None

if __name__ != '__main__':
    # use a clever color palette, eg http://seaborn.pydata.org/api.html#color-palettes
    sns.set(context="talk", style="whitegrid", palette="hls", font="sans-serif", font_scale=0.95)

    # Use tex by default http://matplotlib.org/2.0.0/users/dflt_style_changes.html#math-text
    # mpl.rcParams['text.usetex'] = True  # XXX force use of LaTeX
    mpl.rcParams['font.family'] = "sans-serif"
    mpl.rcParams['font.sans-serif'] = "DejaVu Sans"
    mpl.rcParams['mathtext.fontset'] = "cm"
    mpl.rcParams['mathtext.rm'] = "serif"

    # Configure size for axes and x and y labels
    # Cf. https://stackoverflow.com/a/12444777/
    mpl.rcParams['axes.labelsize']  = "x-small"
    mpl.rcParams['xtick.labelsize'] = "x-small"
    mpl.rcParams['ytick.labelsize'] = "x-small"
    mpl.rcParams['figure.titlesize'] = "x-small"

    # Configure the DPI of all images, once for all!
    mpl.rcParams['figure.dpi'] = DPI
    # print(" - Setting dpi of all figures to", DPI, "...")  # DEBUG

    # Configure figure size, even of if saved directly and not displayed, use HD screen
    # cf. https://en.wikipedia.org/wiki/Computer_display_standard
    mpl.rcParams['figure.figsize'] = FIGSIZE
    # print(" - Setting 'figsize' of all figures to", FIGSIZE, "...")  # DEBUG

def prepare_file_name(file_name = None, alg_name = None, ext_format = None, add_timestamp=True):    
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d-%H-%M-%S")
    
    cwd = os.getcwd() # current directory
    target_directory = "{}\{}".format(cwd, "results")
    
    if not os.path.exists(target_directory):
        try:
            os.makedirs(target_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    file_name_no_ext = ""    
    file_path = "" 
    if alg_name is None and add_timestamp == False:   
        file_name_no_ext = "{}".format(file_name if file_name is not None else "-")    
        
        file_path = "{}\{}.{}".format(target_directory, file_name_no_ext,
                     ext_format if ext_format is not None else "")
    else:
        file_name_no_ext = "{}-{}-{}".format(file_name if file_name is not None else "", 
                 alg_name if alg_name is not None else "", 
                 current_date if add_timestamp else "")  
        
        file_path = "{}\{}.{}".format(target_directory, file_name_no_ext,
                     ext_format if ext_format is not None else "")  
        
    
    return file_path, file_name_no_ext

def read_data_frame(file_name, ext_format='pkl'):
    """ 
    Read a DataFrame from the default path with file name identified as 'file_name'
    """
    file_path, file_name = prepare_file_name(file_name=file_name, ext_format=ext_format, add_timestamp=False)
    df = pd.read_pickle(file_path) 
    
    return df

def make_palette(nbColors, hls=HLS, viridis=False):
    """ 
    Use the seaborn palette to create nbColors different curves on the same figure.
    See also http://seaborn.pydata.org/generated/seaborn.hls_palette.html#seaborn.hls_palette
    """
    if viridis:
        return sns.color_palette('viridis', nbColors)
    else:
        return sns.hls_palette(nbColors + 1)[:nbColors] if hls else sns.husl_palette(nbColors + 1)[:nbColors]


def make_markers(nbMarkers):
    """ 
    Give a list of cycling markers. See also https://matplotlib.org/3.1.1/api/markers_api.html
    List of markers in SMPyBandits (as an example):
        allmarkers = ['o', 'D', 'v', 'p', '<', 's', '^', '*', 'h', '>']

    """
    allmarkers = ['o', 'D', 'v', 'X', 'P', '^', 'p', '<', 's', '^', '*', 'h', '>']
    marker_list = allmarkers * (1 + int(nbMarkers / float(len(allmarkers))))  # Cycle the good number of time
    return marker_list[:nbMarkers]  # Truncate


#: Shrink factor if the legend is displayed on the right of the plot.
SHRINKFACTOR = 0.60

#: Default parameter for maximum number of label to display in the legend INSIDE the figure
MAXNBOFLABELINFIGURE = 8

def display_legend(putatright=False, fontsize="xx-small", shrinkfactor=SHRINKFACTOR, 
           maxnboflabelinfigure=MAXNBOFLABELINFIGURE, fig=None, title=None):
    """plt.legend() with good options, cf. http://matplotlib.org/users/recipes.html#transparent-fancy-legends.
    - For the purpose of generating figures for papers, it is not recommended to place it at the right-side.
    """
    try:
        len_leg = len(plt.gca().get_legend_handles_labels()[1])
        putatright = len_leg > maxnboflabelinfigure
        if len_leg > maxnboflabelinfigure: 
            print("Warning: forcing to use putatright = {} because there is {} items in the legend.".format(putatright, len_leg))  # DEBUG
    except (ValueError, AttributeError, IndexError) as e:
        print("error =", e)  # DEBUG
    
    if fig is None:
        fig = plt
    if putatright:
        try:
            # Shrink current axis by 20% on xaxis and 10% on yaxis
            delta_rect = (1. - shrinkfactor)/6.25
            fig.tight_layout(rect=[delta_rect, delta_rect, shrinkfactor, 1 - 2*delta_rect])
            # Put a legend to the right of the current axis
            fig.legend(loc='center left', numpoints=1, fancybox=True, framealpha=0.8, bbox_to_anchor=(1, 0.5), title=title, fontsize=fontsize)
        except:
            fig.legend(loc='best', numpoints=1, fancybox=True, framealpha=0.8, title=title, fontsize=fontsize)
    else:
        fig.legend(loc='best', numpoints=1, fancybox=True, framealpha=0.8, title=title, fontsize=fontsize)
        

def plot_data_frame(input_dframe, xlabel, ylabel, huelabel, stylelabel=None, height=5, aspect=1.25, flag_semilogx=False,
                    save_file_name=None, sav_file_ext=None, save_data_name=None):
    """
    plot_data_frame() takes 'input_dframe' as the payload data. \
    It also tries to plot the repeated simulation results with the labels of x, y axis and 
    the huelabel identified by the keys of 'input_dframe' as 'xlabel', 'ylabel' and 'huelabel'.
    """
#    sns.set(font_scale=1.0)
    sns_figure = sns.relplot(x=xlabel, y=ylabel, hue = huelabel, style=stylelabel,
                kind="line", data=input_dframe, height=height, aspect=aspect);
    
    if flag_semilogx == True:
        sns_figure.ax.set(xscale="log")         
        
    # force scientific notations on x-axis
    formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    formatter_func = lambda x,pos : "${}$".format(formatter._formatSciNotation('%1.10e' % x))
    
    sns_figure.ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(formatter_func))
    sns_figure.ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(formatter_func))
                 
    if save_file_name is not None:
        sav_file_ext = sav_file_ext if sav_file_ext is not None else 'pdf'        
        figure_file_path, figure_file_name = prepare_file_name(file_name=save_file_name, ext_format=sav_file_ext)
        sns_figure.savefig(figure_file_path)
        
    data_file_name = None
    if save_data_name is not None:
        data_file_path, data_file_name = prepare_file_name(file_name=save_data_name, ext_format='pkl', add_timestamp=True)
        input_dframe.to_pickle(data_file_path)        
           
    return sns_figure, data_file_name


"""
Specifically used for plotting regret data, with theoretical bound
"""
def plot_repeated_simu_results(start, horzion, nbPoints, 
                        nbArm=2, c1=100, c2=20, flag_bound = False,
                        key_x='Total number of plays', key_y='Average regret', key_alg='Algorithms',
                        data_file_name='regret_data', save_fig_name="monte_carlo_regret"):
    #plot key_x, key_y with huelable as key_alg
    repeated_play_data = read_data_frame(data_file_name)   
    
    if flag_bound:
        T = np.linspace(start=4*start, stop = horzion, num=nbPoints)
        
        # This formula is heuristic, and for different parameter sets (context-arm numbers)
        # we need to obtain the proper parameters of a tight bound with manually testing.
        X = (c2*nbArm*(np.log2(T+2)**2) + c1*nbArm*np.log2(T+2))/T 
        Label = ['$O(M\log_2^{\delta}(T))$']*len(T)
        
        Dash = [1]*len(T)
        
        T = np.append(repeated_play_data[key_x], T)
        X = np.append((repeated_play_data[key_y]), X)
        Label = np.append((repeated_play_data[key_alg]), Label)
        Dash = np.append([0]*len(repeated_play_data[key_alg]), Dash)
    
        recorded_data = {}            
        recorded_data[key_x] = T        
        recorded_data[key_y] = X    
        recorded_data[key_alg] = Label
        recorded_data['Dash'] = Label
        
        final_data = pd.DataFrame(recorded_data)    
        
        g, data_file_name = plot_data_frame(final_data, xlabel=key_x, ylabel=key_y, huelabel=key_alg)
        
        nbLines = len(set(final_data[key_alg]))
        print(nbLines)
        
#        # force scientific notations on x-axis
#        g.ax.get_xaxis().get_major_formatter().set_scientific(True)
        g.ax.lines[nbLines-1].set_linestyle("--")
        g.ax.lines[nbLines-1].set_color("grey")
        
        le = g.ax.legend()
        le.get_lines()[nbLines].set_color('grey')
        le.get_lines()[nbLines].set_linestyle("--")
        le.get_frame().set_facecolor('none')    
        le.get_frame().set_edgecolor('none')    
    else:
        final_data = repeated_play_data    
        g, data_file_name = plot_data_frame(final_data, xlabel=key_x, ylabel=key_y, huelabel=key_alg)
#        # force scientific notations on x-axi
#        g.ax.get_xaxis().get_major_formatter().set_scientific(True)
        
    # force scientific notations on x-axis
    formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    formatter_func = lambda x,pos : "${}$".format(formatter._formatSciNotation('%1.10e' % x))
    
    g.ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(formatter_func))
    g.ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(formatter_func))   
          
    file_path, file_name = prepare_file_name(file_name=save_fig_name, ext_format='pdf', add_timestamp=False)
    g.savefig(file_path)  