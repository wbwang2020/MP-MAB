# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:31:54 2019

@author: wenbo2017
"""

"""
Partially inspired by the project SMPyBandits. This file defines the running framework of the bandit simulation.
This file defines the reward generation and collision resolution method "collision_models".
to be extended to other types of collisions, currently only the non-colliding player is rewarded with non-zero value
"""

__author__ = "Wenbo Wang"

import numpy as np

def onlyRewardNoCollision(t, arms, players, choices, pulls, collisions):
    """ Simple collision model where only the players alone on one arm samples it and receives the reward.
    
    - The numpy array 'choices' is the choices of players choosing arms
    - Collision should be rewarded 0
    """

    nb_collisions = np.bincount(choices, minlength=len(arms))

    for i, player in enumerate(players):  # Loop over the player set
        # pulls counts the number of selection, not the number of successful selection.
        pulls[i, choices[i]] += 1
        if nb_collisions[choices[i]] <= 1:  # No collision
            player.getReward(choices[i])  # Observing reward
        else:
            collisions[choices[i]] += 1  # Should be counted here, onlyUniqUserGetsReward
            # handleCollision_or_getZeroReward(player, choices[i])  # NOPE
            player.getCollisionReward(choices[i])
            
# Default collision model to use
defaultCollisionModel = onlyRewardNoCollision


#: List of possible collision models
collision_models = [
    onlyRewardNoCollision,
]