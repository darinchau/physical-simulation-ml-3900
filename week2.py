## This module is code from week one. We try to keep everything outside

from train import train_mlp, train_gaussian_process
from load import load_data_week_1, make_anim
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

def get_inputs(n):
    inputs = np.arange(101)*0.75/100
    data = load_data_week_1()
    train_idx = list(range(n))
    return inputs, data, train_idx

# Define the neural network
class WeekTwoNet(nn.Module):
    def __init__(self):
        super(WeekTwoNet, self).__init__()
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

def train_week_2():
    net = WeekTwoNet()
    inputs, data, train_idx = get_inputs(10)
    train_mlp(inputs, data, train_idx, net, model_name = "model_week2")

def dem1():
    pred = predict_with_name("model_week2")
    make_anim(pred, "w2_pred1.gif")

    data = load_data_week_1()
    make_anim(data, "w2_data.gif")

    err = np.abs(pred - data)
    make_anim(err, "w2_err1.gif")

def dem2():
    inputs, data, train_idx = get_inputs(10)

    model, _ = train_gaussian_process(inputs, data, train_idx)

    idxs = inputs.reshape((-1, 1))
    pred = model.predict(idxs)
    pred = pred.reshape((101, 129, 17))

    err = np.abs(pred - data)

    make_anim(pred, "w2_pred2.gif")
    make_anim(err, "w2_err2.gif")

def train_gaussian_with_first_n_data(n):
    inputs, data, train_idx = get_inputs(n)

    model, (mse, worse) = train_gaussian_process(inputs, data, train_idx)

    idxs = inputs.reshape((-1, 1))
    pred = model.predict(idxs)
    pred = pred.reshape((101, 129, 17))

    return mse, worse


if __name__ == "__main__":
    hist = []
    idxs = []
    for i in range(1, 101):
        print(f"Using the first {i} data - ", end = "")
        mse, worst = train_gaussian_with_first_n_data(i)
        hist.append((mse, worst))
        idxs.append(i)

    plt.figure()
    plt.plot(idxs, hist)
    plt.yscale('log')
    plt.legend(['MSE', 'Worst error'])
    plt.show()