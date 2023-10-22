from NN import NeuralNetwork
import numpy as np
from torch.utils.data import Dataset, DataLoader
import  torch
import torch.optim as optim
class CustomDataset(Dataset):
    def __init__(self, npz_file):
        loaded_data = np.load(npz_file)
        self.data = [loaded_data[key] for key in loaded_data]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = self.data[idx]
        features = torch.tensor(sample[:-1], dtype=torch.float32)  # Features as float
        label = torch.tensor(sample[-1], dtype=torch.float32)  # Label as float
        return features, label
if __name__ == '__main__':
    #############################################################################################
    """Traing of the network"""
    dataset = CustomDataset('./Datas/data_9644.npz')
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = NeuralNetwork()
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(data_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.20f}')
    torch.save(model.state_dict(), f'./Trained_models/{dataset.__len__()}.pth')