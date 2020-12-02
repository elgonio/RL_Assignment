from AbstractAgent import AbstractAgent
#import nle
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces

import torch
import torch.nn as nn

import random
from collections import deque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyAgent(AbstractAgent):
    
    def __init__(self, observation_space, action_space,**kwargs):
        self.filepath ="/workspace/REINFORCE_AGENT_policy_state_dict.pth"
        self.learning_rate = 1e-4
        self.gamma = 1
        self.observation_space = observation_space
        self.action_space = action_space

        stats_observation_space = self.observation_space["blstats"]
        map_observation_space = self.observation_space["glyphs"]
        # for example, if your agent had a Pytorch model it must be load here
        # from Policy import Policy
        self.policy_model = Policy(map_observation_space,stats_observation_space,self.action_space)
        self.policy_model.load_state_dict(torch.load( self.filepath, map_location=torch.device(device)))
        self.optimiser = torch.optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)
        self.log_probs = []

        

        # raise NotImplementedError

    def act(self, observation):
        # Perform processing to observation

        # TODO: return selected action
        map = observation["glyphs"]
        stats = observation["blstats"]

        x,y = stats[0], stats[1]

        n = 3
        s = 2
        padded_map = np.pad(map,n,'edge')
        aleph = padded_map[y-s+n:y+s+n+1,x-s+n:x+s+n+1].flatten()
        aleph = torch.from_numpy(aleph).float().to(device)
        aleph = aleph.unsqueeze(0)

        map = torch.from_numpy(map).float().to(device)
        map = map.unsqueeze(0)
        map = map.unsqueeze(1)
        stats = observation["blstats"]
        stats = torch.from_numpy(stats).float().to(device)
        stats= stats.unsqueeze(0)
        probs = self.policy_model.forward(map, stats, aleph)

        # choose actions according to probability
        
        p = probs.cpu().detach().numpy().flatten()
        action = np.random.choice(self.action_space.n, p=p)
        self.log_probs.append(torch.log(probs[action]))

        return action


    def update(self,rewards):
        self.optimiser.zero_grad()
        returns = self.compute_returns_naive_baseline(rewards)

        policy_gradient = []
        for i in range(len(returns)):
            # why is this negative?
            policy_gradient.append(-self.log_probs[i] * returns[i])
        policy_gradient = torch.stack(policy_gradient).sum()

        # back propagation
        policy_gradient.backward()
        self.optimiser.step()
        self.log_probs = []
        

    def compute_returns(self,rewards):
        returns = []
        for t in range(len(rewards)): 
            G = 0
            for k in range(t,len(rewards)):
                G = G + rewards[k] * self.gamma**(k-t-1)
            returns.append(G)

        return returns

    def compute_returns_naive_baseline(self,rewards):
        baseline = np.mean(rewards)
        std = np.std(rewards)

        returns = []
        for t in range(len(rewards)):
            G = 0
            for k in range(t,len(rewards)):
                # G = G + rewards[k] * gamma**(k-t-1)
                G = G + rewards[k]/(std+0.01) * self.gamma**(k-t-1) 
                # maybe try normalizing with returns
            
            G = G - baseline
            
            returns.append(G)
        return returns

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


