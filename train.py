import numpy as np
from load_mesh import load_data, wrap_data, make_anim, index_exclude, index_include
from models import *
import matplotlib.pyplot as plt
import time

model_name = "model_2"

def train_mlp():
    data, device = load_data()
    inputs = np.arange(101)
    train_idx = list(range(101))[::10]
    ins, outs = wrap_data(inputs, data, train_idx)

    # Train on MLP
    net, hist = train_mlp1(ins, outs, epochs = 200000, verbose = False)

    hist = hist[1:]

    # Plot the training data
    plt.figure()
    plt.plot(hist)
    plt.yscale('log')
    plt.show()

    print(min(hist, key = lambda x: x[1]))

    # Save the model
    torch.save(net, model_name)
    return net, data

def predict(net):
    # net = torch.load(model_name)
    # # net = torch.load(f"{model_name}.pt")
    # net.eval()

    recreated_data = np.zeros((101, 129, 17))

    with torch.no_grad():
        for i in range(101):
            inp = torch.tensor([i*0.75/100]).float()
            recreated_data[i,:,:] = net(inp).numpy()[0]

    return recreated_data

def predict_with_name(name):
    net = torch.load(name)
    # net = torch.load(f"{model_name}.pt")
    net.eval()
    return predict(net)

def train_gaussian_process():
    # Reshape data because gaussian process expects one dimensional output only
    data, _ = load_data()
    data = data.reshape((101, -1))

    inputs = np.arange(101)*0.75/100

    # Create and split training data and testing data via index
    idxs = np.arange(101)
    training_idx = list(range(101))[::10]

    train_idx, test_idx = index_include(idxs, training_idx), index_exclude(idxs, training_idx)

    # Create the data

    xtrain, xtest = inputs[train_idx], inputs[test_idx]
    ytrain, ytest = data[train_idx], data[test_idx]

    xtrain = xtrain.reshape(-1, 1)
    xtest = xtest.reshape(-1, 1)

    # Train the model
    model, err = train_gaussian1(xtrain, xtest, ytrain, ytest)

    return model, data, err
