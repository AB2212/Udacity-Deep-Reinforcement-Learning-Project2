import numpy as np
import matplotlib.pyplot as plt

def init_layers(layer):
    
    """
    Function to calculate the initialization
    values for the weights of the network,
    based on the DDPG paper
    https://arxiv.org/abs/1509.02971
    
    """
    
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    
    return (-lim, lim)

def plot_reward(scores, 
                filename = 'reward_plot.png'):

    plt.plot( scores )
    plt.title("Reward at each episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.savefig(filename)
    plt.show()
    



