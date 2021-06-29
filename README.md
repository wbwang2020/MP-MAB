# Contextual Multi-Player Multi-Armed Bandit (MP-MAB)
## Purpose of This Package:
This repository contains the Python codes for the numerical simulations of the following paper (we also aim to extend this package with more MP-MAB algorithms in the future):

[[Wang2020](http://arxiv.org/abs/2003.13314)] Wenbo Wang, Amir Leshem, Dusit Niyato and Zhu Han, "Decentralized Learning for Channel Allocation in IoT Networks over Unlicensed Bandwidth as a Contextual Multi-player Multi-armed Bandit Game".

This project currently implements the multi-player multi-armed bandit (MP-MAB) algorithms proposed in the following (preprint) papers:

[[Wang2020](http://arxiv.org/abs/2003.13314)] Wenbo Wang, Amir Leshem, Dusit Niyato and Zhu Han, "Decentralized Learning for Channel Allocation in IoT Networks over Unlicensed Bandwidth as a Contextual Multi-player Multi-armed Bandit Game".

[[Rosenski2016](http://proceedings.mlr.press/v48/rosenski16.pdf)] Rosenski, J., Shamir, O., & Szlak, L. (2016, June). Multi-player bandits–a musical chairs approach. In International Conference on Machine Learning (pp. 155-163).

[[Bistritz2018](https://papers.nips.cc/paper/7952-distributed-multi-player-bandits-a-game-of-thrones-approach)] Bistritz, I., & Leshem, A. (2018). Distributed multi-player bandits-a game of thrones approach. In Advances in Neural Information Processing Systems (pp. 7222-7232).

[[Boursier2019](https://hal.archives-ouvertes.fr/hal-02371008/)] E. Boursier and V. Perchet, “Sic-mmab: synchronisation involves communication in multiplayer multi-armed bandits,” in Advances in Neural Information Processing Systems, Vancouver CANADA, Dec. 2019, pp. 12 071–12 080

[[Sumit2019](https://ieeexplore.ieee.org/document/8792108)] Sumit J. Darak and Manjesh K. Hanawal, "Multi-player multi-armed bandits for stable allocation in # heterogeneous ad-hoc networks", IEEE JSAC oct. 2019.

[[Tibrewal2019](https://arxiv.org/abs/1901.03868)] Tibrewal, H., Patchala, S., Manjesh K. Hanawal and Sumit J. Darak (2019). "Multiplayer multiarmed bandits for optimal assignment in heterogeneous networks," arXiv preprint arXiv:1901.03868.

**Note**: [Tibrewal2019] is provided as an alternative algorithm for comparison with [Sumit2019].

## Main Structure
This Python package contains the following modules:

1. Simulation entrances: main_MPMAB.py and main_MPMAB_IoT_Simu.py

2. Configurations for simulation: simu_config.py   
  - We suggest the simulations to be run in the parallel mode for a small size of network and non-parallel mode for larger network (due to possible memory limits).

3. Simulation engine: GameEvaluator.py   

  - Environment generator:
    - MPMAB.py: MAB environment simulator for known distribution of the arm values,
    - HetNetSimulator.py: for a home-brewed IoT HetNet environment,
    - Arms.py: arm simulator for known distribution.
  - Algorithm organizer:
    - MABAlgorithms.py: implementations of the centralized Hugarian algorithm, [Wang2020] (TrialandError), [Rosenski2016] (MusicalChairs) and [Bistritz2018] (GameofThrone),
    - MABAlgorithms2.py: implementations of [Sumit2019] (SOC),
    - MABAlgorithms2a.py: implementations of[Tibrewal2019] (ESE, cf. SOC).
    - MABAlgorithms3.py: implementation of [Boursier2019] (SIC-MMAB)
  - Player simulator:
    - Players.py, Players2.py, Players2a.py and Players3.py: the player simulators corresponding to different MABAlgorithms*.py files
  - Result recorder: PlayResult.py

4. Miscellaneous utilities: plotutils.py, envutils.py, loggingutils.py
