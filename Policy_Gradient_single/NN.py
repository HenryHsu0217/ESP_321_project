import torch
import torch.nn as nn
import torch.nn.functional as F
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define the hidden layers
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        
        # Define the output layers for the mean and standard deviation
        self.mean_head = nn.Linear(128, 1)
        self.std_dev_head = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        mean = torch.tanh(self.mean_head(x)) * 3 
        std_dev = F.softplus(self.std_dev_head(x)) 
        return mean, std_dev