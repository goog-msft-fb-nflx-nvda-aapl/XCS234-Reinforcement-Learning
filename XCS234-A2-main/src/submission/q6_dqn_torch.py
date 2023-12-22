import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import get_logger, join
from utils.test_env import EnvTest
from .q3_schedule import LinearExploration, LinearSchedule
from .q5_linear_torch import Linear

import yaml

yaml.add_constructor("!join", join)

config_file = open("config/q6_dqn.yml")
config = yaml.load(config_file, Loader=yaml.FullLoader)

############################################################
# Problem 6: Implementing DeepMind's DQN
############################################################


class NatureQN(Linear):
    """
    Implementation of DeepMind's Nature paper, please consult the methods section of the paper linked below for details on model configuration.
    (https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)
    """

    ############################################################
    # Problem 6a: initialize_models

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The in_channels 
        to Conv2d networks will n_channels * self.config["hyper_params"]["state_history"]

        Args:
            q_network (torch model): variable to store our q network implementation

            target_network (torch model): variable to store our target network implementation

        TODO:
             (1) Set self.q_network to the architecture defined in the Nature paper associated to this question.
                Padding isn't addressed in the paper but here we will apply padding of size 2 to each dimension of
                the input to the first conv layer (this should be an argument in nn.Conv2d).
            (2) Set self.target_network to be the same configuration self.q_network but initialized from scratch
            (3) Be sure to use nn.Sequential in your implementation.

        Hints:
            (1) Start by figuring out what the input size is to the networks.
            (2) Simply setting self.target_network = self.q_network is incorrect.
            (3) The following functions might be useful
                - nn.Sequential (https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
                - nn.Conv2d (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
                - nn.ReLU (https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
                - nn.Flatten (https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html)
                - nn.Linear (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n

        ### START CODE HERE ###
        # Define the Q network architecture from the Nature paper
        """
        if The input to the neural network consists of an 84 X 84 X 4 image produced by the preprocessing map w.
        """
        self.q_network = nn.Sequential(
            # The first hidden layer convolves 32 filters of 8 X 8 with stride 4 with the input image and applies a rectifier nonlinearity.
            nn.Conv2d(in_channels=n_channels * self.config["hyper_params"]["state_history"], out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            # out_height = 1 + floor ( (in_height - kernel_height + 2 * padding_height ) / stride_height )
            # out_height = 1 + math.floor ( (84 - 8 + 2 * 2 ) / 4 )  = 21
            
            # The second hidden layer convolves 64 filters of 4 X 4 with stride 2, again followed by a rectifier nonlinearity.
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            # out_height = 1 + floor ( (in_height - kernel_height + 2 * padding_height ) / stride_height )
            # out_height = 1 + math.floor ( (21 - 4 + 2 * 0 ) / 2 )  = 9
            
            # Tensorflow : conv3 = layers.conv2d(conv2, num_outputs=64, kernel_size=(3, 3), stride=1, activation_fn=tf.nn.relu)
            # This is followed by a third convolutional layer that convolves 64 filters of 3 X 3 with stride 1 followed by a rectifier.
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            # out_height = 1 + floor ( (in_height - kernel_height + 2 * padding_height ) / stride_height )
            # out_height = 1 + math.floor ( (9 - 3 + 2 * 0 ) / 1 )  = 7
            
            # Tensorflow : full_inputs = layers.flatten(conv3)
            # The nn.Flatten() layer then flattens the spatial dimensions, resulting in a 1D tensor.
            nn.Flatten(),
            
            # The final hidden layer is fully-connected and consists of 512 rectifier units.
            # in_features = num_channels × height × width = 64 x 7 x 7 = 3136
            
            nn.Linear(in_features=3136, out_features=512),
            nn.ReLU(),
            
            # The output layer is a fully-connected linear layer with a single output for each valid action.
            nn.Linear(in_features=512, out_features=num_actions)
        )

        # Initialize the target network with exactly the SAME configuration
        self.target_network = nn.Sequential(
            nn.Conv2d(in_channels=n_channels * self.config["hyper_params"]["state_history"], out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )
        ### END CODE HERE ###

    ############################################################
    # Problem 6b: get_q_values

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state (torch tensor): shape = (batch_size, img height, img width, nchannels x config["hyper_params"]["state_history"])
            network (str): The name of the network, either "q_network" or "target_network"

        Returns:
            out (torch tensor): shape = (batch_size, num_actions)

        TODO:
            Perform a forward pass of the input state through the selected network and return the output values.


        Hints:
            (1) You can forward a tensor through a network by simply calling it (i.e. network(tensor))
            (2) Look up torch.permute (https://pytorch.org/docs/stable/generated/torch.permute.html)
        """
        out = None

        ### START CODE HERE ###
        
        """
        select the network based on the input argument
        network (str): The name of the network, 
            either "q_network" 
            or "target_network"
        """
        if network == "q_network":
            selected_network = self.q_network
        else:
            selected_network = self.target_network

        
        """
        1. we know that state (torch tensor) has a shape of (batch_size, img height, img width, nchannels x config["hyper_params"]["state_history"])
            0 - B : batch_size
            1 - H : img height
            2 - W : img width
            3 - nchannels x config["hyper_params"]["state_history"]
        2. but the input state has dimensions ( batch_size, channels * history, height, width )
            0 - B : batch_size
            3 - nchannels x config["hyper_params"]["state_history"]
            1 - H : img height
            2 - W : img width
        3. so we need to use torch.permute (https://pytorch.org/docs/stable/generated/torch.permute.html) to change the order.
        """ 
        state = state.permute(0, 3, 1, 2)

        """
        Perform a forward pass of the input state through the selected network "selected_network" and return the output values "out".
        We can forward a tensor through a network by simply calling it (i.e. selected_network(state) )
        """
        out = selected_network(state)
        ### END CODE HERE ###

        return out
