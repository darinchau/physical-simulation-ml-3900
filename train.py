import numpy as np
from load import load_data_week_1, wrap_data, make_anim, index_exclude, index_include
import matplotlib.pyplot as plt
import time
import torch
from torch import nn
import numpy as np
import torch.optim as optim
from tqdm import trange
from lazypredict.Supervised import LazyRegressor
from raw_train import GaussianRegression

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
    model, err = GaussianRegression().fit(inputs, data, training_idx, verbose = True)

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
