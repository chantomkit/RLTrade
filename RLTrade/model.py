import math
import random

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_all(self):
        """Get all experiences stored in the memory."""
        return list(self.memory)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """A simple feedforward neural network for DQN.

    Args:
        n_observations (int): Number of observations.
        n_actions (int): Number of actions.
        hidden_layers (list): List of hidden layer sizes.
    """

    def __init__(self, n_observations, n_actions, hidden_layers=[32]):
        super(DQN, self).__init__()
        layer_dims = [n_observations] + hidden_layers
        self.hidden = nn.ModuleList(
            [
                nn.Linear(layer_dims[i], layer_dims[i + 1])
                for i in range(len(layer_dims) - 1)
            ]
        )
        self.output = nn.Linear(layer_dims[-1], n_actions)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.output(x)
