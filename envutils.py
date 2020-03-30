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

# This file defines the class Struct used in simu_config.py, 
# and the automation method for arm parameter generation

__author__ = "Wenbo Wang"

import numpy as np

if __name__ == '__main__':
    print("Warning: this script 'envutils.py' is NOT executable..")  # DEBUG
    exit(0)


class Struct(object):
    """
    Simple class for instantiating objects to add arbitrary attributes as variables.
    Used for serializing configurations parameters.
    Reference:
        https://stackoverflow.com/questions/6198372/most-pythonic-way-to-provide-global-configuration-variables-in-config-py/43941592
    """
    def __init__(self, *args):
        self.__header__ = str(args[0]) if args else None

    def __repr__(self):
        if self.__header__ is None:
             return super(Struct, self).__repr__()
        return self.__header__

    def next(self):
        """ Fake iteration functionality.
        """
        raise StopIteration

    def __iter__(self):
        """ Fake iteration functionality.
        We skip magic attribues and Structs, and return the rest.
        """
        ks = self.__dict__.keys()
        for k in ks:
            if not k.startswith('__') and not isinstance(k, Struct):
                yield getattr(self, k)

    def __len__(self):
        """ Don't count magic attributes or Structs.
        """
        ks = self.__dict__.keys()
        return len([k for k in ks if not k.startswith('__')\
                    and not isinstance(k, Struct)])
        

def uniform_means(nbContext=2, nbPlayers=2, nbArms=4, delta=0.05, lower=0., upper=1.):
    """
    Return a dictionary of lower and upper bounds of arm values, 
    well spaced (needed for some algorithms that requires arm-values to be distrigushed) for uniform distribution:

    - in [lower, upper],
    - starting from lower + (upper-lower) * delta, up to lower + (upper-lower) * (1 - delta),
    - and there is nbArms arms.

    >>> np.array(uniformMeans(2, 0.1))
    array([0.1, 0.9])
    >>> np.array(uniformMeans(3, 0.1))
    array([0.1, 0.5, 0.9])
    >>> np.array(uniformMeans(9, 1 / (1. + 9)))
    array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    """
    assert nbPlayers >= 1, "Error: 'nbPlayers' = {} has to be >= 1.".format(nbPlayers)  # DEBUG
    assert nbArms >= 1, "Error: 'nbArms' = {} has to be >= 1.".format(nbArms)  # DEBUG
    assert nbArms >= nbPlayers, "Error: 'nbArms' has to be larger than 'nbPlayers'."
    assert upper - lower > 0, "Error: 'upper - lower' = {:.3g} has to be > 0.".format(upper - lower)  # DEBUG
    assert 0. < delta < 1., "Error: 'delta' = {:.3g} has to be in (0, 1).".format(delta)  # DEBUG
    mus = lower + (upper-lower) * np.linspace(delta, 1 - delta, nbArms)

    means = [];
    for idPlayer in range(nbPlayers):
        np.random.shuffle(mus)
        means.append(mus)
    return means


def randomMeans(nbPlayers=2, nbArms=4, mingap=None, lower=0., upper=1.):
    """Return a list of means of arms, randomly sampled uniformly in [lower, lower + amplitude], with a min gap >= mingap.

    - All means will be different, except if ``mingap=None``, with a min gap > 0.

    """
    assert nbArms >= 1, "Error: 'nbArms' = {} has to be >= 1.".format(nbArms)  # DEBUG
    assert upper - lower > 0, "Error: 'upper - lower' = {:.3g} has to be > 0.".format(upper - lower)  # DEBUG
    mus = np.random.rand(nbArms)
    if mingap is not None and mingap > 0:
        assert (nbArms * mingap) < (upper - lower / 2.), "Error: 'mingap' = {:.3g} is too large, it might be impossible to find a vector of means with such a large gap for {} arms.".format(mingap, nbArms)  # DEBUG
        
        means = []
        for idPlayer in range(nbPlayers):            
            while np.min(np.abs(np.diff(mus))) <= mingap:  # Ensure a min gap > mingap
                mus = np.random.rand(nbArms)
            
            mus = lower + (upper - lower) * mus
            means.append(mus)
   
    return means