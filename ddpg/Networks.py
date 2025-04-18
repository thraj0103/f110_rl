import torch
import torch.nn as nn
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.Relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.Relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, output_size)
        self.tanh = nn.Tanh()

        self.steering_low  = -0.4
        self.steering_high =  0.4
        self.vel_low       = 0.1
        self.vel_high      =  5.0


    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.Relu1(x1)
        x3 = self.fc2(x2)
        x4 = self.Relu2(x3)
        x5 = self.fc3(x4)
        raw_action = self.tanh(x5)  # [-1, 1] per dimension

        raw_steering = raw_action[:, 0]
        raw_velocity = raw_action[:, 1]

        steering = self._rescale(raw_steering, self.steering_low, self.steering_high)
        velocity = self._rescale(raw_velocity, self.vel_low, self.vel_high)

        if torch.isnan(raw_action).any():
            torch.set_printoptions(threshold=float('inf'))
            print("==== NaN detected in mu network output ====")
            print("Input to policy network:", x)
            print("fc1 output:", x1)
            print("fc2 output:", x3)
            print("fc3 output (before tanh):", x5)
            print("Final output (after tanh):", raw_action)

        return torch.stack([steering, velocity], dim=1)
    
    def _rescale(self, x, low, high):
         x = low + (x+1) * (high - low) / 2.0
         return x
    

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
    def forward(self, x):
        x = self.net(x)
        return x