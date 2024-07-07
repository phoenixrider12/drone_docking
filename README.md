# Offshore Drone Docking using Deep-RL
## [Project Website](https://phoenixrider12.github.io/DroneDocing) | [Paper](https://www.sciencedirect.com/science/article/pii/S1568494624006173?ref=pdf_download&fr=RR-2&rr=89f668a94f1b919e)
This is the codebase for our paper [Deep Reinforcement Learning for Sim2Real Policy Transfer in VTOL-UAVs Offshore Docking Operations](https://www.sciencedirect.com/science/article/pii/S1568494624006173?ref=pdf_download&fr=RR-2&rr=89f662ca6e01919e). We present a framework for docking operation of VTOL-UAVs in offshore docking operations using deep reinforcement learning

# Installation
We have implemented two RL algorithms, Deep Q-Networks(DQN) and Proximal Policy Optimization(PPO). We provide the code for both of them.

```
git clone https://github.com/phoenixrider12/drone_docking.git
cd drone_docking
conda create -n docking pip
conda activate docking
pip install -r requirements.txt
```

To run DQN:
```
git checkout DQN
python main.py
```

To run PPO:
```
git checkout PPO
python main.py
```

# Citation
If you find our work useful for your research, please cite:
```
@article{ALI2024111843,
title = {Deep Reinforcement Learning for sim-to-real policy transfer of VTOL-UAVs offshore docking operations},
journal = {Applied Soft Computing},
volume = {162},
pages = {111843},
year = {2024},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2024.111843},
url = {https://www.sciencedirect.com/science/article/pii/S1568494624006173},
author = {Ali M. Ali and Aryaman Gupta and Hashim A. Hashim},
keywords = {Unmanned Aerial Vehicles, Offshore docking, Reinforcement learning, Deep Q learning, Proximal Policy Optimization, Sim2Real},
abstract = {This paper proposes a novel Reinforcement Learning (RL) approach for sim-to-real policy transfer of Vertical Take-Off and Landing Unmanned Aerial Vehicle (VTOL-UAV). The proposed approach is designed for VTOL-UAV landing on offshore docking stations in maritime operations. VTOL-UAVs in maritime operations encounter limitations in their operational range, primarily stemming from constraints imposed by their battery capacity. The concept of autonomous landing on a charging platform presents an intriguing prospect for mitigating these limitations by facilitating battery charging and data transfer. However, current Deep Reinforcement Learning (DRL) methods exhibit drawbacks, including lengthy training times, and modest success rates. In this paper, we tackle these concerns comprehensively by decomposing the landing procedure into a sequence of more manageable but analogous tasks in terms of an approach phase and a landing phase. The proposed architecture utilizes a model-based control scheme for the approach phase, where the VTOL-UAV is approaching the offshore docking station. In the Landing phase, DRL agents were trained offline to learn the optimal policy to dock on the offshore station. The Joint North Sea Wave Project (JONSWAP) spectrum model has been employed to create a wave model for each episode, enhancing policy generalization for sim2real transfer. A set of DRL algorithms have been tested through numerical simulations including value-based agents and policy-based agents such as Deep Q Networks (DQN) and Proximal Policy Optimization (PPO) respectively. The numerical experiments show that the PPO agent can learn complicated and efficient policies to land in uncertain environments, which in turn enhances the likelihood of successful sim-to-real transfer.}
}
```
