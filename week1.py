## This module is code from week one. We try to keep everything outside

from train import train_mlp, train_gaussian_process
from load import load_data_week_1, make_anim
import numpy as np
import torch
from torch import nn

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
    data = load_data_week_1()
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
    data = load_data_week_1()
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