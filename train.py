import numpy as np
from load import load_data_week_1, wrap_data, make_anim, index_exclude, index_include
import matplotlib.pyplot as plt
import time
import torch
from torch import nn
import numpy as np
import torch.optim as optim
from tqdm import trange
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

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

    print(min(hist, key = lambda x: x[1]))

    if model_name is None:
        model_name = f"model{time.time()}"

    # Save the model
    torch.save(net, model_name)
    return net, data, hist

# Raw code for gaussian model training
def train_gaussian_raw(xtrain, xtest, ytrain, ytest):
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

    print(f"Finished training Gaussian process. Error: {mse}, worse: {np.max(np.abs(ytest - ypred))}")

    return model, mse

# Wrapper around raw code
def train_gaussian_process(inputs, data, training_idx):
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

    # Train the model
    model, err = train_gaussian_raw(xtrain, xtest, ytrain, ytest)

    return model, err
