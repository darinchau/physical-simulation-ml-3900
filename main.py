from train import train_mlp, predict_with_name, predict, train_gaussian_process
from load_mesh import load_data, make_anim
import numpy as np
from load_mesh import index_include

def dem1():
    a = predict_with_name("model_1")
    make_anim(a)

    data, _ = load_data()
    make_anim(data)

if __name__ == "__main__":
    # model, data = train_gaussian_process()
    # ins =
    idxs = np.arange(101)
    train_idx = list(range(101))[::10]
    id = index_include(idxs, train_idx)
    print(id)

    ins = (np.arange(101)*0.75/100)
    print(ins[id])