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

__author__ = "Wenbo Wang"

#from random import random
from numpy.random import random as nprandom

import scipy.stats as stats

class Arm(object):
    """ Base class for an arm class."""

    def __init__(self, param):
        """ Base class for an arm class."""
        self.lower = param["lower_val"]  #: Lower value of rewardd, array[context]
        self.upper = param["upper_val"]  #: Upper value of rewards
        self.amplitude = self.upper - self.lower  #: Amplitude of value of rewards
        
        # for arm of a specific context-player
        self.context = param["context"]
        self.playerID = param["playerID"]
        self.armID = param["armID"]
        
        # prepare samples
        self.horizon = 0
        self.prepared_samples = []
        
    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def lower_amplitude(self):
        """(lower, amplitude)"""
        if hasattr(self, 'lower') and hasattr(self, 'amplitude'):
            return self.lower, self.amplitude
        else:
            raise NotImplementedError("This method lower_amplitude() has to be implemented in the class inheriting from Arm.")
            
    @property
    def current_context(self):
        """(lower, amplitude)"""
        if hasattr(self, 'context_set'): 
            return self.context
        else:
            raise NotImplementedError("This method current_context() has to be implemented in the class inheriting from Arm.")

    # --- Printing

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.__dir__)

    # --- Random samples

    def draw_sample(self, t=None):
        """ Draw one random sample."""
        raise NotImplementedError("This method draw_sample(t) has to be implemented in the class inheriting from Arm.")  
        
    def prepare_samples(self, horizon):
        raise NotImplementedError("This method prepare_samples(horizon) has to be implemented in the class inheriting from Arm.")  
        
"""
Uniform distribution arms
"""
class UniformArm(Arm):
    """ Uniformly distributed arm, default in [0, 1],
    """

    def __init__(self, param):
        """New arm."""
        self.lower = param["lower_val"]  #: Lower value of rewardd, array[context]
        self.upper = param["upper_val"]  #: Upper value of rewards
        self.amplitude = self.upper - self.lower  #: Amplitude of value of rewards
        if self.amplitude <= 0:
            raise Exception("The upper bound must be larger than the lower bound")
        
        self.mean = (self.lower + self.upper) / 2.0  #: Mean for this UniformArm arm
        self.variance = self.amplitude**2 / 12.0 #: Variance for ths UniformArm arm
        
        self.context = param["context"]
        self.playerID = param["playerID"]
        self.armID = param["armID"]
        
        # prepare samples
        self.horizon = 0
        self.prepared_samples = []

    # --- Random samples

    def draw_sample(self, context, t=None):
        """ Draw one random sample."""
        if self.context != context:
            raise Exception("the arm corresponding to a different context is called")
        
        if t is None:
            # The parameter t is ignored in this Arm. Do sampling right away.
            return self.lower + (nprandom() * self.amplitude)
        else:
            if t >= self.horizon:
                raise Exception("the time instance is beyond the horizon")
            else:
                return self.prepared_samples[t]

    def prepare_samples(self, horizon):
        if horizon <= 0:
            raise Exception("the input horizon is invalid")
        else:
            self.horizon = horizon
            self.prepared_samples = self.lower + (nprandom(self.horizon) * self.amplitude)

    # --- Printing

    def __str__(self):
        return "UniformArm"

    def __repr__(self):
        return "U({:.3g}, {:.3g})".format(self.lower, self.upper)

"""
Gaussian distribution arms
"""
class GaussianArm(Arm):
    """ 
    Gaussian distributed arm, possibly truncated.
    - The default setting is to truncate into [0, 1] (so Gaussian.draw() is sampled in [0, 1]).
    """

    def __init__(self, param):
        """New arm."""
        self.mu = param["mu"]
        if "sigma" not in param.keys():
            self.sigma = 0.05
        else:
            self.sigma = param["sigma"]
            assert self.sigma > 0, "The parameter 'sigma' for a Gaussian arm has to be > 0."
        
        self.lower = 0# used to truncate the sampled value
        self.upper = 1# used to truncate the sampled value
        
        # For the trunctated normal distribution, see:
        # "Simulation of truncated normal variables", https://arxiv.org/pdf/0907.4010.pdf
        # Section "Two-sided truncated normal distribution"
        
        alpha = (self.lower - self.mu) / self.sigma
        beta = (self.upper - self.mu) / self.sigma
        
        self.sampler = stats.truncnorm(alpha, beta, loc=self.mu, scale=self.sigma)
        
        self.mean, self.variance = self.sampler.stats(moments='mv')
        
        
        self.context = param["context"]
        self.playerID = param["playerID"]
        self.armID = param["armID"]
    
    
    # --- Random samples

    def draw_sample(self, context, t=None):
        """ 
        Draw one random sample. The parameter t is ignored in this Arm.
        """
        if self.context != context:
            raise Exception("the arm corresponding to a different context is called")
        
        if t is None:
            # The parameter t is ignored in this Arm. Do sampling right away.
            return self.sampler.rvs(1)
        else:
            if t >= self.horizon:
                raise Exception("the time instance is beyond the horizon")
            else:
                return self.prepared_samples[t]

    def prepare_samples(self, horizon):
        """
        The runcated normal distribution takes a lot more time for giving a single sample each time
        We could pre-sample an array and then retrive them with index of t.
        """
        if horizon <= 0:
            raise Exception("the input horizon is invalid")
        else:
            self.horizon = horizon
            self.prepared_samples = self.sampler.rvs(self.horizon)

    # --- Printing
    def __str__(self):
        return "Gaussian"

    def __repr__(self):
        return "N({:.3g}, {:.3g})".format(self.mean, self.sigma)

"""
Other types of distribution should be implemented here.
"""

if __name__ == '__main__':
    print("Warning: this script 'Arms.py' is NOT executable..")  # DEBUG
    exit(0)

__all__ = ["UniformArm", "GaussianArm"]