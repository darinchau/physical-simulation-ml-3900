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
from tqdm import trange

def train(train_loader, test_loader, net, epochs = 5, verbose = False):
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # Define the loss function
    criterion = nn.MSELoss()

    history = []

    # Train the network
    for epoch in trange(epochs):
        running_loss = 0.0
        for i, dat in enumerate(train_loader, 0):
            inputs, labels = dat
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

        train_loss = running_loss/len(train_loader)

        # Test the network
        with torch.no_grad():
            running_loss = 0.0
            for i, dat in enumerate(test_loader, 0):
                inputs, labels = dat
                inputs = inputs.float()
                labels = labels.float()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            if verbose:
                print("Test loss: %.3f" % (running_loss/len(test_loader)))
            test_loss = running_loss/len(test_loader)

        history.append((train_loss, test_loss))
    return net, history

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(1, 50),
                nn.Linear(50, 129 * 17)
            )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 129, 17) # Reshape output to (batch_size, 129, 17)
        return x


def train_mlp1(train_loader, test_loader, epochs, verbose = False):
    # Instantiate the network and the optimizer
    net = Net()
    return train(train_loader, test_loader, net, epochs, verbose = verbose)
