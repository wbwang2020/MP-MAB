# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:15:17 2019

Partially inspired by the project SMPyBandits. This file defines the running framework of the bandit simulation.
"""

""" 
Uniformly distributed arm in [0, 1], or [lower, upper]_context, for each context.

Example of creating an arm:

>>> import random; import numpy as np
>>> random.seed(0); np.random.seed(0)
>>> Unif01 = UniformArm(0, 1)
>>> Unif01
U(0, 1)
>>> Unif01.mean
0.5

Examples of sampling from an arm:

>>> Unif01.draw()  # doctest: +ELLIPSIS
0.8444...
>>> Unif01.draw_nparray(20)  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
array([0.54... , 0.71..., 0.60..., 0.54..., 0.42... ,
       0.64..., 0.43..., 0.89...  , 0.96..., 0.38...,
       0.79..., 0.52..., 0.56..., 0.92..., 0.07...,
       0.08... , 0.02... , 0.83..., 0.77..., 0.87...])
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Wenbo Wang"
__version__ = "0.6"

from random import random
from numpy.random import random as nprandom

# Local imports
try:
    from .Arm import Arm
except ImportError:
    from Arm import Arm


class UniformArm(Arm):
    """ Uniformly distributed arm, default in [0, 1],

    - default to (mini, maxi),
    - or [lower, lower + amplitude], if (lower=lower, amplitude=amplitude) is given.

    >>> arm_0_1 = UniformArm()
    >>> arm_0_10 = UniformArm(0, 10)  # maxi = 10
    >>> arm_2_4 = UniformArm(2, 4)
    >>> arm_m10_10 = UniformArm(-10, 10)  # also UniformArm(lower=-10, amplitude=20)
    """

    def __init__(self, lower=0., upper=1., context_set):
        """New arm."""
        self.lower = lower  #: Lower value of rewards, corresponding to array of states
        self.upper = upper  #: Upper value of rewards
        self.amplitude = upper - lower  #: Amplitude of value of rewards
        self.context_set = context_set
        
        self.amplitude = upper - lower  #: Amplitude of rewards
        self.mean = (self.lower + self.upper) / 2.0  #: Mean for this UniformArm arm

    # --- Random samples

    def draw(self, t=None):
        """ Draw one random sample. The parameter t is ignored in this Arm."""
        shape = (1, len(self.context_set))
        return self.lower + (nprandom(shape) * self.amplitude)

    # --- Printing

    def __str__(self):
        return "UniformArm"

    def __repr__(self):
        return "U({:.3g}, {:.3g})".format(self.lower, self.upper)

    # --- Lower bound

    @staticmethod
    def kl(x, y):
        """ The kl(x, y) to use for this arm."""
        return klBern(x, y)

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for UniformArm arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / klBern(mu, mumax)


__all__ = ["UniformArm"]


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)