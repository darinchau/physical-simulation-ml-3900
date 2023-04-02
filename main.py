import numpy as np
from load_mesh import load_data, wrap_data
from models import *

def do_stuff():
    data, device = load_data()
    inputs = np.arange(101)
    train_idx = list(range(101))[::10]
    ins, outs = wrap_data(inputs, data, train_idx)

    # Train on MLP
    train_mlp1(ins, outs, (129, 17))
    train_mlp2(ins, outs, (129, 17))
    train_mlp3(ins, outs, (129, 17))

if __name__ == "__main__":
    do_stuff()