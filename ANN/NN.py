import torch
import numpy as np
"""Skeleton of the neural network, having 3 hidden layers with 128, 64, 32 neurons each."""
"""The input is the 8 observable parameter returned from the step function of the env."""
"""And the first 2 activation functionis the ReLu function, and the last one is Tanh function."""
"""All these could be adjusted or revised among the project, for just NN approach, we'll then find a way to collect data set for it to train."""
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(8, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,1)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
"""Testing with the forward of the network with random imput"""
if __name__ == '__main__':
    input=torch.rand([1,8])
    print(input)
    network=NeuralNetwork()
    print(network.forward(input))
    for name, param in network.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")