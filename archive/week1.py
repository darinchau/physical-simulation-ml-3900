## This module is code from week one. We try to keep everything outside

from models import train_mlp, train_gaussian_process
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from load import load_elec_potential, wrap_data, index_exclude, index_include
import matplotlib.pyplot as plt
import time
import numpy as np
import torch.optim as optim
from tqdm import trange
from lazypredict.Supervised import LazyRegressor
from models import GaussianRegression
from matplotlib import animation

def wrap_data(ins, data, train_idx: tuple[int, ...]):
    num_data = len(data)
    class WrappedData(Dataset):
        def __init__(self, input, data, indices):
            self.idxs = np.array(indices)
            self.input = np.array(input)
            self.data = np.array(data)

        def __getitem__(self, index):
            idx = self.idxs[index]
            x = self.input[idx]
            y = self.data[idx]
            return x, y

        def __len__(self):
            return len(self.idxs)

    # Train data from 1, 11, 21, ..., 101
    train_data = WrappedData(ins, data, train_idx)

    # Test data from the others
    test_idx = tuple(set(range(num_data)) - set(train_idx))
    test_data = WrappedData(ins, data, test_idx)

    # Wrap in data loaders
    train_dl = DataLoader(train_data, batch_size=1, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=1, shuffle=False)

    return train_dl, test_dl

def make_anim(data, path = None):
    # Set up figure and axis for animation
    fig, ax = plt.subplots()
    heatmap = ax.imshow(data[0], cmap="hot")

    # Add a colorbar to the heatmap
    cbar = ax.figure.colorbar(heatmap, ax=ax)
    cbar.ax.set_ylabel("Intensity", rotation=-90, va="bottom")

    # Define update function for animation
    def update(frame):
        heatmap.set_data(data[frame])
        return heatmap,

    # Create animation object and display it
    anim = animation.FuncAnimation(fig, update, frames=data.shape[0], interval=50, blit=True)

    if path is not None:
        writergif = animation.PillowWriter(fps=30)
        anim.save(path, writer=writergif)
    else:
        plt.show()

# Raw training code
def train_net(train_loader, test_loader, net, epochs = 5, verbose = False):
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

# Trains the MLP. Returns net, data, history
def train_mlp(inputs, data, training_idx, net, model_name = None):
    train_dl, test_dl = wrap_data(inputs, data, training_idx)

    # Train on MLP
    net, hist = train_net(train_dl, test_dl, net, epochs = 200000, verbose = False)

    hist = hist[1:]

    # Plot the training data
    plt.figure()
    plt.plot(hist)
    plt.yscale('log')
    plt.show()

    min_training = min(enumerate(hist), key = lambda x: x[1][1])

    print(f"Minimum training data: epoch {min_training[0]} with MSE = {min_training[1][1]}")

    if model_name is None:
        model_name = f"model{time.time()}"

    # Save the model
    torch.save(net, model_name)
    return net, data, hist

# Wrapper around raw code
def train_gaussian_process(inputs, data, training_idx):
    # Train the model
    model, err = GaussianRegression().fit_model(inputs, data, training_idx, verbose = True)

    return model, err

# Use the lazypredict library to see if we can get anything out of it
def lazy_predict(inputs, data, training_idx):
    # Reshape data because gaussian process expects one dimensional output only
    num_data = data.shape[0]

    # Make a copy
    data = np.array(data)
    data = data.reshape((num_data, -1))

    # Create and split training data and testing data via index
    idxs = np.arange(num_data)

    train_idx, test_idx = index_include(idxs, training_idx), index_exclude(idxs, training_idx)

    # Create the data
    xtrain, xtest = inputs[train_idx], inputs[test_idx]
    ytrain, ytest = data[train_idx], data[test_idx]

    xtrain = xtrain.reshape(-1, 1)
    xtest = xtest.reshape(-1, 1)

    # Initialize LazyRegressor
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

    # Train and test various models
    models, predictions = reg.fit(xtrain, xtest, ytrain, ytest)

    # Print the model performances
    print(models)

    return models

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

def predict(net):
    recreated_data = np.zeros((101, 129, 17))

    with torch.no_grad():
        for i in range(101):
            inp = torch.tensor([i*0.75/100]).float()
            recreated_data[i,:,:] = net(inp).numpy()[0]

    return recreated_data

def predict_with_name(name):
    net = torch.load(name)
    net.eval()
    return predict(net)

def train_week_1():
    net = Net()
    inputs = np.arange(101)*0.75/100
    data = load_elec_potential()
    train_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    train_mlp(inputs, data, train_idx, net)

# This does not work anymore. Download a snapshot from github of week 1
# def dem1():
#     pred = predict_with_name("./Datas/Week 1/model_1")
#     make_anim(pred, "pred1.gif")

#     data = load_data_week_1()
#     make_anim(data, "data.gif")

#     err = np.abs(pred - data)
#     make_anim(err, "err1.gif")

def dem2():
    inputs = np.arange(101)*0.75/100
    data = load_elec_potential()
    train_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    model, _ = train_gaussian_process(inputs, data, train_idx)

    idxs = inputs.reshape((-1, 1))
    pred = model.predict(idxs)
    pred = pred.reshape((101, 129, 17))

    err = np.abs(pred - data)

    make_anim(pred, "pred2.gif")
    make_anim(data)
    make_anim(err, "err2.gif")

if __name__ == "__main__":
    dem2()