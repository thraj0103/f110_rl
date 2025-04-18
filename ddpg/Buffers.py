import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory:

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        # Ensure all outputs are PyTorch tensors on correct device
        obs_batch = torch.stack([item[0].detach().cpu() for item in batch])                   # tensor [B, obs_dim]
        action_batch = torch.tensor([item[1][0] for item in batch], dtype=torch.float32)      # tensor [B, act_dim]
        reward_batch = torch.tensor([item[2] for item in batch], dtype=torch.float32).unsqueeze(1)  # tensor [B, 1]
        next_obs_batch = torch.stack([item[3].detach().cpu() for item in batch])              # tensor [B, obs_dim]
        done_batch = torch.tensor([float(item[4]) for item in batch], dtype=torch.float32).unsqueeze(1)  # tensor [B, 1]

        # Send to device
        obs_batch = obs_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)
        next_obs_batch = next_obs_batch.to(device)
        done_batch = done_batch.to(device)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10

    def length(self):
        return len(self.memory)