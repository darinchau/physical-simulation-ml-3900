import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from torch.utils.data import Dataset
import torch.optim as optim

def predict_from_sample_input(train_loader, test_loader):
    # Define the neural network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc = nn.Linear(1, 129*17) # Input size is 1, output size is 129*17

        def forward(self, x):
            x = self.fc(x)
            x = x.view(-1, 129, 17) # Reshape output to (batch_size, 129, 17)
            return x

    # Instantiate the network and the optimizer
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # Define the loss function
    criterion = nn.MSELoss()

    # Train the network
    num_epochs = 10
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
        print("Test loss: %.3f" % (running_loss/len(test_loader)))