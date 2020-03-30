#MP-MAB

This repository contains the Python codes for the numerical simulations of the following paper:

[Wang2020] Wenbo Wang, Amir Leshem, Dusit Niyato and Zhu Han, "Decentralized Learning for Channel Allocation in IoT Networks over Unlicensed Bandwidth as a Contextual Multi-player Multi-armed Bandit Game".

This project currently implements the multi-player multi-armed bandit (MP-MAB) algorithms proposed in the following papers:

[Wang2020] Wenbo Wang, Amir Leshem, Dusit Niyato and Zhu Han, "Decentralized Learning for Channel Allocation in IoT Networks over Unlicensed Bandwidth as a Contextual Multi-player Multi-armed Bandit Game".

[Rosenski2016] Rosenski, J., Shamir, O., & Szlak, L. (2016, June). Multi-player bandits–a musical chairs approach. In International Conference on Machine Learning (pp. 155-163).

[Bistritz2018] Bistritz, I., & Leshem, A. (2018). Distributed multi-player bandits-a game of thrones approach. In Advances in Neural Information Processing Systems (pp. 7222-7232).

[Sumit2019] Sumit J. Darak and Manjesh K. Hanawal, "Multi-player multi-armed bandits for stable allocation in # heterogeneous ad-hoc networks", IEEE JSAC oct. 2019.

This Python package contains the following modules:

1. Simulation entrances: main_MPMAB.py and main_MPMAB_IoT_Simu.py
2. Simulation engine: GameEvaluator.py
3. Environment generator: MPMAB.py (for known distribution) and HetNetSimulator.py (for a home-brewed IoT HetNet environment)
4. Algorithm organizer: MABAlgorithms.py and MABAlgorithms2.py
5. Player simulator: Players.py and Players2.py
6. Arm simulator for known distribution: Arms.py
Result recorder: PlayResult.py
Miscellaneous utilities: plotutils.py, envutils.py, loggingutils.py
Configurations for simulation: simu_config.py
