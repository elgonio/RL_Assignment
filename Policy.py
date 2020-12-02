from AbstractAgent import AbstractAgent
import nle
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces

import torch
import torch.nn as nn

import random
from collections import deque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, observation_space_map: spaces.Box, observation_space_stats: spaces.Box, action_space: spaces.Discrete):
        super().__init__()
        self.input_shape = observation_space_map.shape
        self.n_actions = action_space.n

    
        self.conv = nn.Sequential(
            nn.Conv2d(torch.from_numpy(np.array([1])).to(device), 32, kernel_size=4, stride=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU()
        ).to(device)

        # conv_out_size = self._get_conv_out(self.input_shape)
        
        # print("shape", self.input_shape)
        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(self.input_shape[0],4,3),3,2),2,1)
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(self.input_shape[1],4,3),3,2),2,1) 
        conv_out_size = convw*convh*64

        stats_size = observation_space_stats.shape[0]
        # print("convw", self.conv2d_size_out(self.input_shape[0],8,4), "convh", convh)
        # print("conv_out_size", conv_out_size, "stats_size", stats_size)

        # conv_out_size = self._get_conv_out((1,1,21,79))
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + stats_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_actions),
            nn.Sigmoid(),
            nn.Softmax()
        ).to(device)
        
        self.fc_aleph = nn.Sequential(
            nn.Linear(conv_out_size + stats_size + 25, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Linear(256, self.n_actions),
            nn.Sigmoid(),
            nn.Softmax()
        ).to(device)

    def forward(self, x1, x2, x3=None):
        conv_out = self.conv(x1).to(device).view(x1.size()[0], -1)
        # print(conv_out.shape, x2.shape)
        if type(x3) is type(None):
            conv_and_stats = torch.cat((conv_out,x2),1)
            return self.fc(conv_and_stats)
        else:
            # print(conv_out.shape, x2.shape, x3.shape)
            conv_and_stats_and_aleph = torch.cat((conv_out,x2,x3),1)
            return self.fc_aleph(conv_and_stats_and_aleph).flatten()

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(*shape))
        return int(np.prod(o.size()))

    def conv2d_size_out(self, size, kernel_size, stride):
        return ((size - kernel_size) // stride)  + 1