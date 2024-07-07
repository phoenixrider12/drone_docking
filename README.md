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
