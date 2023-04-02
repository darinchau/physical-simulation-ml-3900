import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from torch.utils.data import Dataset
import torch.optim as optim
from load_mesh import split_data
from lazypredict.Supervised import LazyRegressor

def train(train_loader, test_loader, net, verbose = False):
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # Define the loss function
    criterion = nn.MSELoss()

    # Train the network
    num_epochs = 10
    best_test_loss = 999
    for _ in range(3):
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.float()
                labels = labels.float()
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if verbose:
                print("Epoch %d, loss: %.3f" % (epoch+1, running_loss/len(train_loader)))

        # Test the network
        with torch.no_grad():
            running_loss = 0.0
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data
                inputs = inputs.float()
                labels = labels.float()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            if running_loss/len(test_loader) < best_test_loss:
                best_test_loss = running_loss/len(test_loader)
            if verbose:
                print("Test loss: %.3f" % (running_loss/len(test_loader)))
    print(best_test_loss)
    return net, best_test_loss


def train_mlp1(train_loader, test_loader, data_size):
    # Define the neural network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc = nn.Sequential(
                    nn.Linear(1, 50),
                    nn.Linear(50, data_size[0] * data_size[1])
                )

        def forward(self, x):
            x = self.fc(x)
            x = x.view(-1, data_size[0], data_size[1]) # Reshape output to (batch_size, 129, 17)
            return x

    # Instantiate the network and the optimizer
    net = Net()
    return train(train_loader, test_loader, net)

def train_mlp2(train_loader, test_loader, data_size):
    # Define the neural network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc = nn.Sequential(
                    nn.Linear(1, data_size[0] * data_size[1]),
                )

        def forward(self, x):
            x = self.fc(x)
            x = x.view(-1, data_size[0], data_size[1]) # Reshape output to (batch_size, 129, 17)
            return x

    # Instantiate the network and the optimizer
    net = Net()
    return train(train_loader, test_loader, net)

def train_mlp3(train_loader, test_loader, data_size):
    # Define the neural network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc = nn.Sequential(
                    nn.Linear(1, 10),
                    nn.Linear(10, 100),
                    nn.Linear(100, data_size[0] * data_size[1]),
                )

        def forward(self, x):
            x = self.fc(x)
            x = x.view(-1, data_size[0], data_size[1]) # Reshape output to (batch_size, 129, 17)
            return x

    # Instantiate the network and the optimizer
    net = Net()
    return train(train_loader, test_loader, net)