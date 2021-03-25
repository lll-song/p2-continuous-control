import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size, action_size, leak=0.01, seed=42):
        """ Initialize parameters and build model.

        Params
        ======
            state_size (int): State dimension
            action_size (int): Action dimension
            random_seed (int): Random seed
            leak: Leakiness in leaky relu
        """
        super(Actor, self).__init__()
        self.leak = leak
        self.random_seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, action_size)
        self.initialize_weights()

    def initialize_weights(self):
        """ Initilaize weights by He initialization"""
        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.output.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        """Actor (policy) network"""
        state = self.bn0(state)
        x = F.leaky_relu(self.fc1(state), negative_slope=self.leak)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leak)
        x =  torch.tanh(self.output(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_size, action_size, leak=0.01, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): State dimension
            action_size (int): Action dimension
            random_seed (int): Random seed
            fc1_units (int): First hidden layer nodes
            fc2_units (int): second hidden layer nodes
            hidden_size(int): Hidden layers node
        """
        super(Critic, self).__init__()
        self.leak = leak
        self.random_seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128+action_size, 128)
        self.output = nn.Linear(128, 128)        
        self.initialize_weights()


    def initialize_weights(self):
        """ Initilaize weights by He initialization"""
        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.output.weight.data, -3e-3, 3e-3)


    def forward(self, state, action):
        """ Critic (value) network"""
        state = self.bn0(state)
        xs = F.leaky_relu(self.fc1(self.bn0(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        return self.output(x)