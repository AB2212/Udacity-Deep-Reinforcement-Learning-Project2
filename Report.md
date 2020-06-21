## Project 2: Continuous Control

Author: Abhijeet Biswas

### Project Goal

Train an agent to move its double-jointed arm to a goal location and keep it there.

![](/images/reacher.gif)

### Project details

The environment is based on [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents). 

The Unity Machine Learning Agents Toolkit (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. 

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible. We have 20 such agents in this environment that get trained in parallel to share experiences and speed up learning.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.


### Learning Algorithm

I have implemented [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971) in a parallel environment setting (20 agents). 

DDPG is a model-free algorithm based on deterministic policy gradients that operate on high-dimensional continuous action spaces. It consists of Actor and Critic which learns from off-policy data. Similar to Deep Q Network (DQN), it also uses Replay Buffer and Target networks with soft updates to stabilize the training. Total 4 networks are used, 2 (main and target) for actor and critic each. 

The actor function $\mu(s|\theta^{\mu})$ maps the states into actions in a deterministic manner, and the critic $Q(s,a| \theta^{Q})$ takes states and actions as input and outputs Q values. The critic is updated using the Bellman equation as in Q-learning (shown in the image below).
The actor takes in the state and outputs action, this state and action are fed to the critic. The actor is updated in such a way that the action of the actor maximizes the expected Q value output by the critic for that state. This is evaluated using the chain rule as shown in the actor update step in the image below.

Ornstein-Uhlenbeck process is used to generate temporally correlated noise. This noise is added to the action output of the actor for exploration.

Following is the complete algorithm:
![](/images/ddpg_algo.PNG)
image source: [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING](https://arxiv.org/abs/1509.02971)


### CODE

The code is written in Python 3.6.3 and uses PyTorch 0.4.0. I have used detailed comments to explain all the steps.

### Network Architecture

- I have used  4  neural networks (Actor Main, Actor Target, Critic Main & Critic target), where each network has 4 layers.
- For Actors, there are 33(states), 128, 128 and 4(actions) neurons in each layer respectively.
- For Critic, there are 37 (states + actions), 128, 128 and 1 (Q value) neurons in each layer respectively.
- The first layer takes in the input which is states in case of Actor, and states & actions in case of Critic.The last layer outputs the action for Actor and Q value for critic respectively. Intermediate layers are used to build more complex features from input that will produce better results.
- I have used [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation  and Dropout with 0.1 probability after each layer except the last one. For Actor, I have used tanh activation function in the last layer to bound the actions between -1 and 1.

#### Other Hyperparameters

- tau: Used For soft update of target network, constant value of 1e-3 was used

- BUFFER_SIZE: Replay buffer  size of 1e6 was used

- BATCH_SIZE: Minibatch size of  512  was used

- UPDATE_EVERY: Episode Interval for network update, the network was updated after every 20 episodes     

- NUM_UPDATES: Number of passes over the data at each update step, 10 was used

- Ornstein-Uhlenbeck noise parameters: theta = 0.15, max_sigma = 0.2, min_sigma = 0.05

### Results

The environment got solved in 141 episodes by having average reward (32.16) more than 30 over 100 consecutive episodes. The below plot shows the reward received after each episode

![reward_plot.png](/images/reward_plot.png)

### Future Ideas

1. Use prioritized replay buffer to replay important transitions more frequently, and therefore learn more efficiently.
2. Use [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) for more sample efficient learning.
2. Use [Distributed Distributional Deterministic Policy Gradients](https://openreview.net/pdf?id=SyZipzbCb), [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347), [TD3](https://arxiv.org/abs/1802.09477), [SAC](https://arxiv.org/abs/1812.05905), [PlanNet](https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html) and [Dreamer](https://ai.googleblog.com/2020/03/introducing-dreamer-scalable.html) to compare performance against DDPG
