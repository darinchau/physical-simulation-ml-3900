import numpy as np
from load_mesh import load_data, wrap_data, make_anim
from models import *
import matplotlib.pyplot as plt
import time

model_name = "model_2"

def do_stuff():
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



if __name__ == "__main__":
    a = predict_with_name("model_1")
    print(a.shape)
    make_anim(a)

    data, _ = load_data()
    make_anim(data)