# Team: 404-not-fun
Hi, we are the team 404-not-fun. In the reinforcement learning course (COMME5069) of NTU, we are interested in the representation reinforcement learning so as to be our final project topic.

Here we are going to introduce the details of our project.

## Project: StateLM: GPT-based representation reinforcement learning

The project, named StateLM: GPT-based representation reinforcement learnign, focuses on the domain of representation RL using GPT-based model.

Generative pre-trained transformer-based model (GPT) is a powerful sequence model for natural language processing.

And representation reinforcement learning is a RL mechanism to encode some information from real environment to be useful and utilize its representation to train for an RL agent policy.  

GPT-based representation RL is to represent the state to be a meaningful representation information by GPT model based on state trajectories collected from real world, and then use these useful representation as an input to train RL agent. 

In this work, we cover the problem of hard exploration in reinforcement learning. The problem is the environment information is too complex and too sparse to retrieve. Therefore, we encode the high and complex state space into a meaningful vector with low dimension to represent the state. And then, the model is also used as the policy network for training RL agent.

### The experimental simulation environment
* [Gridworld](https://gymnasium.farama.org/), a discrete environment built from gymnasium
    * 21 discrete spaces
    * 4 action space
* [Metaworld](https://meta-world.github.io/), a continuous environment
    * reach-v2
    * push-v2
    * plate-slide-back-side-v2

### The list of RL algorithms in trial
* **SAC (the best)**
* DDPG
* A2C
* PPO
* DQN

### Start the program
We code different RL algorithms based on GPT. You can switch to another branch to try out. The types of RL algorithms as following list:
* feature/online (our best)
* feature/GPT2
* feature/GPT-nano
* feature/DDPG_MT10
* feature/SAC
* feature/VQ
* test/minGPT_gridworld
* test/minGPTPPO_gridworld

Including feature/GPT2, feature/GPT-nano, feature/DDPG_MT10, feature/SAC, and feature/VQ, please start by `train.py` using the command as below:

`python train.py`

The others, please start by `test.py` using the command as below:

`python test.py`

## Contribution
Affliation: National Taiwan University (NTU)

Team Members:
* Yi Hsin You
    * Department: Computer Science
    * Email: b08902017@ntu.edu.tw
* Hao Ying Cheng
    * Department: Electrical Engineering
    * Email: d11921b14@ntu.edu.tw
