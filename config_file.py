class Config():

    BUFFER_SIZE = int(1e6)     # replay buffer size
    BATCH_SIZE = 512           # minibatch size 
    UPDATE_EVERY = 20          # how often to update the network
    NUM_UPDATES = 10           # Number of passes
    NUM_NEURONS_LAYER1 = 128   # Number of neurons in layer 1 
    NUM_NEURONS_LAYER2 = 128   # Number of neurons in layer 2



    def display(self):
        """Display Configuration values"""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
