import numpy as np
from load_mesh import load_data, wrap_data
from models import predict_from_sample_input

def do_stuff():
    data, device = load_data()
    ins = np.arange(101)
    train_idx = list(range(101))[::10]
    ins, outs = wrap_data(ins, data, train_idx)
    predict_from_sample_input(ins, outs)

if __name__ == "__main__":
    do_stuff()