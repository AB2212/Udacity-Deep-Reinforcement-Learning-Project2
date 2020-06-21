import numpy as  np
#Setting random seed for reproducibility
np.random.seed(0) 
import sys
import random
#Setting random seed
random.seed(0)

import torch
#Setting random seed
torch.manual_seed(0)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agent import Agent
from noise_generator import OUNoise
from config_file import Config
from utils import plot_reward

from unityagents import UnityEnvironment
import numpy as np


# Loading the config
config = Config()
config.display()

########################## STARTING ENVIRONMENT ###################################################
###################################################################################################

# select this option to load version 1 (with a single agent) of the environment
#env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')

# select this option to load version 2 (with 20 agents) of the environment
env = UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64')


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


######################### INITIALIZING AGENT ##########################################################
#######################################################################################################

NUM_NEURONS_LAYER1 = config.NUM_NEURONS_LAYER1  # Number of neurons in layer1
NUM_NEURONS_LAYER2 = config.NUM_NEURONS_LAYER2  # Number of neurons in layer2

# Initializing our agent
agent = Agent(state_size,
              action_size,
              [NUM_NEURONS_LAYER1,
               NUM_NEURONS_LAYER1])

# Initializing Ornstein-Uhlenbeck process noise
noise  = OUNoise(action_size)

########################## TRAINING AGENT ##############################################################
########################################################################################################

episodes = 1000                                        # Max no. of episodes   
scores = []                                            # list containing scores from each episode
scores_window = deque(maxlen=100)

for i_episode in range(1,episodes+1):
    
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    state = env_info.vector_observations               # get the current state
    score = np.zeros(num_agents)                       # scores for each agent
    
    noise.reset()
    
    step = 0

    while True:
        
        action = agent.act(state)                      # select an action
        action = noise.get_action(action, step)        # add noise to action for exploration
        
        env_info = env.step(action)[brain_name]        # send the action to the environment
        
        next_state = env_info.vector_observations      # get the next state
        reward = env_info.rewards                      # get the reward
        done = env_info.local_done                     # see if episode has finished
        agent.step(state, action, reward,              # Update Agent's attributes(Replay buffer/parameters)
                   next_state, done)
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        
        step += 1                                      
         
        if np.any(done):                               # exit loop if episode finished is by any agent
            break
        
    scores_window.append(np.mean(score))               # save most recent score
    scores.append(np.mean(score))                      # save most recent score
    
    sys.stdout.flush()
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    
    
    if np.mean(scores_window) >= 30:
        
        torch.save({'actor_main_network_state_dict' : agent.actor_main.state_dict(),
                    'actor_target_network_state_dict' : agent.actor_target.state_dict(),
                    'critic_main_network_state_dict' : agent.critic_main.state_dict(),
                    'critic_target_network_state_dict' : agent.critic_target.state_dict(),
                    'optimizer_actor_state_dict' : agent.optimizer_actor.state_dict(),
                    'optimizer_critic_state_dict' : agent.optimizer_actor.state_dict()},
                    'agent_state_dict.pt')
        
        if np.mean(scores_window) >= 32:
            
            print("\n Problem Solved!")
            
            break
    
print("Score: {}".format(score))

#Plotting Rewards
plot_reward(scores)

############################## TESTING AGENT ########################################################
#####################################################################################################

# Loading saved parameters to test agent over 100 trials
checkpoint = torch.load('agent_state_dict.pt', map_location= "cpu")
agent.actor_main.load_state_dict(checkpoint['actor_main_network_state_dict'])
agent.critic_main.load_state_dict(checkpoint['critic_main_network_state_dict'])

def test(num_episodes=100):
    
    all_scores = []

    from tqdm import tqdm
    
    for ep in tqdm(range(num_episodes)):

        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        while True:
            actions = agent.act(states)                        # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break

    all_scores.append(np.mean(scores))
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(all_scores)))
    
test(num_episodes = 100)





