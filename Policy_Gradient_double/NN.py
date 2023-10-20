import torch
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mean_net = torch.nn.Sequential(
            torch.nn.Linear(8, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        
        self.std_dev_net = torch.nn.Sequential(
            torch.nn.Linear(8, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Softplus()  
        )
    def forward(self, x):
        mean = self.mean_net(x)
        std_dev = self.std_dev_net(x)
        return mean, std_dev