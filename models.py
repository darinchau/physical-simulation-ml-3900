import torch
from torch import nn
import numpy as np
import torch.optim as optim
from tqdm import trange
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

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

def train_gaussian1(xtrain, xtest, ytrain, ytest):
    # Define the kernel function
    kernel = RBF(length_scale=1.0)

    # Define the Gaussian Process Regression model
    model = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10)

    # Train the model on the training data
    model.fit(xtrain, ytrain)

    # Predict the output for the testing data
    ypred = model.predict(xtest)

    # Calculate the mean square error
    mse = np.mean((ytest - ypred)**2)

    print(f"Finished training Gaussian process. Error: {mse}")

    return model, mse