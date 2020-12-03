# Contextual Multi-Player Multi-Armed Bandit (MP-MAB)
## Purpose of This Package:
This repository contains the Python codes for the numerical simulations of the following paper:

[[Wang2020](http://arxiv.org/abs/2003.13314)] Wenbo Wang, Amir Leshem, Dusit Niyato and Zhu Han, "Decentralized Learning for Channel Allocation in IoT Networks over Unlicensed Bandwidth as a Contextual Multi-player Multi-armed Bandit Game".

This project currently implements the multi-player multi-armed bandit (MP-MAB) algorithms proposed in the following papers:

[[Wang2020](http://arxiv.org/abs/2003.13314)] Wenbo Wang, Amir Leshem, Dusit Niyato and Zhu Han, "Decentralized Learning for Channel Allocation in IoT Networks over Unlicensed Bandwidth as a Contextual Multi-player Multi-armed Bandit Game".

[[Rosenski2016](http://proceedings.mlr.press/v48/rosenski16.pdf)] Rosenski, J., Shamir, O., & Szlak, L. (2016, June). Multi-player bandits–a musical chairs approach. In International Conference on Machine Learning (pp. 155-163).

[[Bistritz2018](https://papers.nips.cc/paper/7952-distributed-multi-player-bandits-a-game-of-thrones-approach)] Bistritz, I., & Leshem, A. (2018). Distributed multi-player bandits-a game of thrones approach. In Advances in Neural Information Processing Systems (pp. 7222-7232).

[[Sumit2019](https://ieeexplore.ieee.org/document/8792108)] Sumit J. Darak and Manjesh K. Hanawal, "Multi-player multi-armed bandits for stable allocation in # heterogeneous ad-hoc networks", IEEE JSAC oct. 2019.

[[Boursier2019](https://papers.nips.cc/paper/2019/file/c4127b9194fe8562c64dc0f5bf2c93bc-Paper.pdf)] E. Boursier and V. Perchet, "Sic-mmab: synchronisation involves communication in multiplayer multi-armed bandits," in Advances in Neural Information Processing Systems, Vancouver CANADA, Dec. 2019, pp. 12 071–12 080.

We also aim to extend this package with more MP-MAB algorithms.

## Main Structure
This Python package contains the following modules:

1. Simulation entrances: main_MPMAB.py and main_MPMAB_IoT_Simu.py
2. Configurations for simulation: simu_config.py   

3. Simulation engine: GameEvaluator.py   
  - Environment generator: MPMAB.py (for known distribution) and HetNetSimulator.py (for a home-brewed IoT HetNet environment)   
    - Arm simulator for known distribution: Arms.py   
  - Algorithm organizer: MABAlgorithms.py, MABAlgorithms2.py, MABAlgorithms3.py
    - Player simulator: Players.py, Players2.py, Players3.py
    - Result recorder: PlayResult.py
4. Miscellaneous utilities: plotutils.py, envutils.py, loggingutils.py
