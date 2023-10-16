import torch
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mean_net = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
        
        self.std_dev_net = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Softplus()  
        )
    def forward(self, x):
        mean = self.mean_net(x)
        std_dev = self.std_dev_net(x)
        return mean, std_dev