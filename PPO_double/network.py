import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
class network(nn.Module):
    def __init__(self,output_dim):
        super(network, self).__init__()
        self.layer1 = nn.Linear(4, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(64, output_dim)
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        activation3 = F.relu(self.layer3(activation2))
        activation4 = F.relu(self.layer4(activation3))
        activation5 = F.relu(self.layer5(activation4))
        output = self.layer6(activation5)
        return output