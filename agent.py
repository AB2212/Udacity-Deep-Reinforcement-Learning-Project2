import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import Actor, Critic
from utils import init_layers
from replay_buffer import ReplayBuffer
from config_file import Config


BUFFER_SIZE = Config.BUFFER_SIZE                # replay buffer size
BATCH_SIZE = Config.BATCH_SIZE                  # minibatch size 
UPDATE_EVERY = Config.UPDATE_EVERY              # how often to update the network
NUM_UPDATES = Config.NUM_UPDATES                # Number of passes

#Device type (use cuda if gpu is available else cpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    
    def __init__(self, state_size, action_size, 
                 hidden_neurons, dropout = 0.1,
                 gamma = 0.99, 
                 learning_rate_actor = 1e-3, 
                 learning_rate_critic = 1e-3,
                 seed = 0, tau = 1e-3):
        
        """
        Arguments: state_size [Int] (State size),
                   action_size [Int] (Number of Actions),
                   hidden_neurons [List] (Neurons in each hidden layer),
                   dropout [Float from 0 to 1] (Dropout Regularization),
                   gamma [Float 0 to 1] (Discounting factor),
                   learning_rate_actor [Float 0 to 1] (Learning rate for weight 
                                                       update for actor),
                   learning_rate_critic [Float 0 to 1] (Learning rate for weight
                                                        update for critic),
                   seed [Int] (random seed),
                   tau [Float from 0 to 1] (Soft update rate for target DQN)
                    
        """
    
        
        super().__init__()
        
        
        # Initializing main actor 
        self.actor_main = Actor(state_size, action_size, 
                              hidden_neurons, dropout).to(device)
        
        # Initializing target actor 
        self.actor_target = Actor(state_size, action_size, 
                              hidden_neurons, dropout).to(device)
        
        # Initializing main critic
        self.critic_main = Critic(state_size, action_size, 
                              hidden_neurons, dropout).to(device)
        
        # Initializing target critic
        self.critic_target = Critic(state_size, action_size, 
                              hidden_neurons, dropout).to(device)
        
        # Optimizer for actor
        self.optimizer_actor = optim.Adam(self.actor_main.parameters(),
                                          lr = learning_rate_actor)
        
        # Optimizer for critic
        self.optimizer_critic = optim.Adam(self.critic_main.parameters(),
                                          lr = learning_rate_critic)
        
        self.gamma = gamma
        
        
        self.tau = tau
        
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
        self.t_step = 0
        
        
        
    def step(self,state, action, reward, next_state, done):
        
        """
        Arguments: state [Numpy Array] (current state),
                   action [Numpy Array] (action taken at current state)
                   reward  (float) (reward received for current action)
                   next_state [Numpy Array] (next state as a result of current action)
                   done (Bool) (Whether episode has end or not)
        
        """
        
        # Adding experience to replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # Keeping track of time step
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        # Updating model weights after 
        # regular intervals
        if self.t_step == 0:
            
            # Checking if enough samples are present
            if len(self.memory) > BATCH_SIZE:
                
                for _ in range(NUM_UPDATES):
                
                    self.learn(self.memory.sample())
                
                
    def act(self, state):
        
        """
        Arguments: state [Torch Tensor] (environment state),
                   
                   
        Returns: Actions [Numpy Array]
        
        """
        
        # Setting the network to evaluation mode
        self.actor_main.eval()
        
        # Using torch no grad as action selection
        with torch.no_grad():
        
            state = torch.from_numpy(state).float().to(device)
            
            action =  self.actor_main(state)
        
            action.clamp_(-1., 1.)
        
        # Setting model for training
        self.actor_main.train()
        
        return action.squeeze(0).cpu().numpy()
    
    
    
    def learn(self,experiences):
        
        """
        Arguments: experience [Tuple]( tuple containing sequence of 
                   state_t, action_t, reward_t, state_t+1, and 
                   each being a Torch Tensor)
                   
        """
        
        # Experiences are fetched from Replay buffer
        states, actions, rewards, next_states, dones = experiences
        

        
        # Setting target network to evaluaion mode
        self.critic_target.eval()
        self.actor_target.eval()
        
        # Creating target value for critic
        with torch.no_grad():
            
            next_actions = self.actor_target(next_states)

            Q_next_state = self.critic_target(next_states, next_actions)
            
            Q_target = rewards + self.gamma*Q_next_state*(1.- dones)
            
            
        # Training Critic
        self.critic_main.train()
        
        # Calculating critic Q value for state and action
        Q_critic = self.critic_main(states, actions)
        
        # Using MSE loss for critic
        critic_loss = F.mse_loss(Q_target, Q_critic).mean()
        
        # Zero grad removes any accumulated gradient
        self.optimizer_critic.zero_grad()
        
        # Calculating gradients using backpropagation
        critic_loss.backward()
        
        # Clipping high gradients
        torch.nn.utils.clip_grad_norm_(self.critic_main.parameters(), 1)
        
        # Updating Weights
        self.optimizer_critic.step()
        
        
        # Calculating critic Q value for state and 
        # actor's action
        actor_loss = -self.critic_main(states,self.actor_main(states)).mean()
        
        # Zero grad removes any accumulated gradient
        self.optimizer_actor.zero_grad()
        
        # Calculating gradients using backpropagation
        actor_loss.backward()
        
        # Clipping high gradients
        torch.nn.utils.clip_grad_norm_(self.actor_main.parameters(), 1)
        
        # Updating Weights
        self.optimizer_actor.step()
        
    
        
        # Updating the target network using soft update through
        # weighted sum of previous parameters and current parameters
        self.soft_update(self.actor_main, self.actor_target)
        
        self.soft_update(self.critic_main, self.critic_target)
        
    
    
                
    def soft_update(self, main_model, target_model):
        
        """
        Arguments: main_model [MLP object]
                   target_model [MLP object]
                   
        Description: updates the weight of the target model 
                     network using weighted sum of previous target model  
                     parameters and current main model parameters 

        """
        
        for target_param, main_param in zip(target_model.parameters(),
                                             main_model.parameters()):
            
            target_param.data.copy_(self.tau * main_param.data + (1.-self.tau) * target_param.data)
    
        
