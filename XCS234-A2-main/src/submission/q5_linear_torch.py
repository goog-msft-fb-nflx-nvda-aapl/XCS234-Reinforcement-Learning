import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.test_env import EnvTest
from utils.general import join
from core.deep_q_learning_torch import DQN
from .q3_schedule import LinearExploration, LinearSchedule

import yaml

yaml.add_constructor("!join", join)

config_file = open("config/q5_linear.yml")
config = yaml.load(config_file, Loader=yaml.FullLoader)

############################################################
# Problem 5: Linear Approximation
############################################################


class Linear(DQN):
    """
    Implementation of a single fully connected layer with Pytorch to be utilized
    in the DQN algorithm.
    """

    ############################################################
    # Problem 5b: initializing models

    def initialize_models(self):
        """
        Creates the 2 separate networks (Q network and Target network).
        The input to these networks will be an image of shape img_height * img_width with
        channels = n_channels * self.config["hyper_params"]["state_history"].

        - self.network (torch model): variable to store our q network implementation
        - self.target_network (torch model): variable to store our target network implementation

        TODO:
            (1) Set self.q_network to be a linear layer with num_actions as the output size.
            (2) Set self.target_network to be the same configuration as self.q_netowrk but initialized by scratch.

        Hint:
            (1) Start by figuring out what the input size is to the networks.
            (2) Simply setting self.target_network = self.q_network is incorrect.
            (3) Consult nn.Linear (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) which should be useful for your implementation.
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n
        
        ### START CODE HERE ###
        # calculate the input size for the linear layer
        channels   = n_channels * self.config["hyper_params"]["state_history"]
        input_size = img_height * img_width * channels
        
        # TODO:(1) Set self.q_network to be a linear layer with num_actions as the output size.
        self.q_network      = nn.Linear(input_size, num_actions)
        
        """
        TODO: (2) Set self.target_network to be the same configuration as self.q_netowrk but initialized by scratch.
        The comment "initialized from scratch" emphasizes that a new instance is created for the target network, 
        ensuring that its parameters are not shared with the Q network.
        In PyTorch, when you create a new instance of a model (e.g., nn.Linear), 
        it initializes the parameters with new random values. 
        This ensures that the networks start with different sets of weights and biases.
        """
        self.target_network = nn.Linear(input_size, num_actions)
        ### END CODE HERE ###

    ############################################################
    # Problem 5c: get_q_values

    def get_q_values(self, state, network="q_network"):
        """
        Returns Q values for all actions.

        Args:
            state (torch tensor): shape = (batch_size, img height, img width, nchannels x config["hyper_params"]["state_history"])
            network (str): The name of the network, either "q_network" or "target_network"

        Returns:
            out (torch tensor): shape = (batch_size, num_actions)

        TODO:
            Perform a forward pass of the input state through the selected network and return the output values.

        Hints:
            (1) Look up torch.flatten (https://pytorch.org/docs/stable/generated/torch.flatten.html)
            (2) Pay attention to the torch.flatten `start_dim` Parameter 
            (3) You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None

        ### START CODE HERE ###
        # Choose the network based on the input argument
        if network == "q_network":
            networkSelected = self.q_network
        else:
            networkSelected = self.target_network

        """
        Hints:
            (1) Look up torch.flatten (https://pytorch.org/docs/stable/generated/torch.flatten.html)
            (2) Pay attention to the torch.flatten `start_dim` Parameter
        The torch.flatten function is used to flatten a multidimensional tensor into a one-dimensional tensor.
        The start_dim parameter specifies the dimension from which flattening should start.
        Setting start_dim=1 means that flattening will start from the second dimension (index 1) of the input tensor.
        
        we know from above description that 
        state (torch tensor): shape = (batch_size, img height, img width, nchannels x config["hyper_params"]["state_history"])
        The tensor state has a shape with four dimensions: 
        (batch_size, img height, img width, nchannels x config["hyper_params"]["state_history"]). 
        
        When flattening, 
        we want to collapse all dimensions starting from the second dimension (img height) into a one-dimensional tensor.
        The reasoning is the following:
            1. The first dimension is the batch dimension (batch_size), and we don't want to flatten it.
            2. The second dimension is img height, and we want to start flattening from this dimension.
            3. The remaining dimensions (img width and nchannels x config["hyper_params"]["state_history"]) will be collapsed into a single dimension.
        
        Therefore, by setting start_dim=1, 
        we ensure that flattening starts from the second dimension (img height) and proceeds accordingly.
        This is crucial for correctly transforming the input tensor into a format suitable for the subsequent linear layers in the neural network.
        """
        state = torch.flatten(state, start_dim=1)

        # Hints (3) You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        out = networkSelected(state)
        ### END CODE HERE ###

        return out

    ############################################################
    # Problem 5d: update_target

    def update_target(self):
        """
        The update_target function will be called periodically to copy self.q_network weights to self.target_network.

        TODO:
            Update the weights for the self.target_network with those of the self.q_network.

        Hint:
            Look up loading pytorch models with load_state_dict function.
            (https://pytorch.org/tutorials/beginner/saving_loading_models.html)
        """

        ### START CODE HERE ###
        self.target_network.load_state_dict( self.q_network.state_dict() )
        ### END CODE HERE ###

    ############################################################
    # Problem 5e: calc_loss

    def calc_loss(
        self,
        q_values: torch.Tensor,
        target_q_values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated_mask: torch.Tensor, 
        truncated_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the MSE loss of a given step. The loss for an example is defined:
            Q_samp(s) = r if terminated or truncated
                        = r + gamma * max_a' Q_target(s', a') otherwise
            loss = (Q_samp(s) - Q(s, a))^2

        Args:
            q_values: (torch tensor) shape = (batch_size, num_actions)
                The Q-values that your current network estimates (i.e. Q(s, a') for all a')

            target_q_values: (torch tensor) shape = (batch_size, num_actions)
                The Target Q-values that your target network estimates (i.e. (i.e. Q_target(s', a') for all a')

            actions: (torch tensor) shape = (batch_size,)
                The actions that you actually took at each step (i.e. a)

            rewards: (torch tensor) shape = (batch_size,)
                The rewards that you actually got at each step (i.e. r)
            
            terminated_mask: (torch tensor) shape = (batch_size,)
                A boolean mask of examples where we reached the terminal state

            truncated_mask: (torch tensor) shape = (batch_size,)
                A boolean mask of examples where the episode was truncated

        TODO:
            Return the MSE loss for a given step. 
            You may use the function description for guidance in your implementation.

        Hint:
            You may find the following functions useful
                - torch.max (https://pytorch.org/docs/stable/generated/torch.max.html)
                - torch.sum (https://pytorch.org/docs/stable/generated/torch.sum.html)
                - torch.bitwise_or (https://pytorch.org/docs/stable/generated/torch.bitwise_or.html)
                - torch.gather:
                    * https://pytorch.org/docs/stable/generated/torch.gather.html
                    * https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms

            You may need to use the variables:
                - self.config["hyper_params"]["gamma"]
        """
        gamma = self.config["hyper_params"]["gamma"]

        ### START CODE HERE ###
        """
        q_samp is calculated according to the Bellman equation.
        It is a tensor containing the target Q values for the selected actions.
        The calculation considers rewards, discount factor (γ), and masks for termination and truncation.
        
        1. Rewards: Tensor containing the rewards obtained at each step. This represents the immediate reward for taking an action in the current state.
        2. Target Q-values:
            target_q_values: Tensor containing the Q-values predicted by the target network for the next state.
            torch.max(target_q_values, dim=1): Along dimension 1, find the maximum value for each example in the batch. This returns a tuple containing two tensors: the maximum values and their corresponding indices.
            torch.max(target_q_values, dim=1).values: Extracts the tensor of maximum values.
        3. Discount Factor and Masks:
            gamma: The discount factor, representing the weight given to future rewards.
            (1 - terminated_mask.float()): A mask that is 1 for examples that are not terminated and 0 for terminated examples.
            (1 - truncated_mask.float()): A mask that is 1 for examples that are not truncated and 0 for truncated examples.
        """
        discounted_max_Q_value = gamma * torch.max(target_q_values, dim=1).values #Calculates the discounted maximum Q-value for the next state.
        discounted_max_Q_value *= (1 - terminated_mask.float()) #Zeros out the discounted maximum Q-value for terminated examples.
        discounted_max_Q_value *= (1 - truncated_mask.float()) #Further zeros out the result for examples that are both terminated and truncated.
        q_samp = rewards +  discounted_max_Q_value #Combines the immediate rewards with the discounted and masked maximum Q-values to compute the target Q-values (q_samp) for the current state.

        """
        1. iput Tensors:
            1-1. q_values: Tensor containing the Q-values predicted by the Q-network for each action in the current state.
            1-2. actions: Tensor containing the indices of the actions that were actually taken.
        2.Reshaping actions:
            2-1. actions.view(-1, 1): Reshapes the actions tensor into a 2D tensor with one column.
            2-2. The -1 is used to infer the size along that dimension, and 1 specifies one column.
            2-3. This reshaping is necessary for the torch.gather function.
        3. Using torch.gather:
            3-1. torch.gather(input, dim, index): Gathers values along a specified axis based on the provided indices.
                3-1-1. input: The input tensor from which values are gathered (in this case, q_values).
                3-1-2. dim: The axis along which to index. In this case, 1 indicates the second dimension (columns).
                3-1-3. index: The tensor containing the indices to be gathered. This is the reshaped actions tensor.
        4. Result q_values_for_actions:
            q_values_for_actions: The resulting tensor contains the Q-values corresponding to the actions that were actually taken. Each row corresponds to an example in the batch, and the values are selected based on the indices provided in the actions tensor.
        """
        q_values_for_actions = torch.gather( q_values, 1, actions.view(-1, 1).long() )

        """
        1. Target Q-values (q_samp):
            q_samp: This tensor represents the target Q-values for the actions taken in a given state.
            It is calculated using the Bellman equation and considers rewards, discount factor (gamma), and masks for termination and truncation.
        2. Reshaping q_samp:
            q_samp.view(-1, 1): Reshapes the q_samp tensor into a 2D tensor with one column. 
            This reshaping is done to align the shape with q_values_for_actions, which is also a 2D tensor with one column.
        3. Predicted Q-values (q_values_for_actions):
            q_values_for_actions: 
                This tensor contains the Q-values predicted by the Q-network for the actions that were actually taken. 
                It is obtained using torch.gather to select the relevant Q-values from the Q-values predicted for all actions.
        4. Calculating MSE Loss:
            F.mse_loss(input, target): 
                Computes the mean squared error between each element in the input tensor and the corresponding element in the target tensor.
                1. input: The tensor containing predicted values (in this case, reshaped q_samp).
                2. target: The tensor containing target values (in this case, q_values_for_actions).
        5. Result loss:
            loss: 
                The resulting scalar value represents the mean squared difference between the predicted Q-values and the target Q-values.
                This loss is used as the objective function to train the Q-network via backpropagation.
        """
        loss = F.mse_loss( q_samp.view(-1, 1), q_values_for_actions )

        return loss
        ### END CODE HERE ###

    ############################################################
    # Problem 5f: add_optimizer

    def add_optimizer(self):
        """
        This function sets the optimizer for our linear network

        TODO:
            Set self.optimizer to be an Adam optimizer optimizing only the self.q_network parameters

        Hint:
            Look up torch.optim.Adam (https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)
            What are the input to the optimizer's constructor?
        """
        ### START CODE HERE ###
        # Set self.optimizer to be an Adam optimizer optimizing only the self.q_network parameters
        self.optimizer = torch.optim.Adam( self.q_network.parameters() )
        ### END CODE HERE ###
