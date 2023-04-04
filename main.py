from train import train_mlp, predict_with_name, predict, train_gaussian_process
from load_mesh import load_data, make_anim
import numpy as np
from load_mesh import index_include

from scipy.optimize import minimize

def dem1():
    pred = predict_with_name("model_1")
    make_anim(pred, "dem1_pred.gif")

    data, _ = load_data()
    make_anim(data, "dem1_data.gif")

    err = np.abs(pred - data)
    err = err / np.max(err)
    make_anim(err, "dem1_err.gif")

def dem2():
    model, data, _ = train_gaussian_process()

    idxs = (np.arange(101)*0.75/100).reshape((-1, 1))
    pred = model.predict(idxs)
    pred = pred.reshape((101, 129, 17))
    make_anim(pred, "dem2_pred.gif")

    data, _ = load_data()
    make_anim(data, "dem2_data.gif")

    err = np.abs(pred - data)
    err = err / np.max(err)
    make_anim(err, "dem2_err.gif")

if __name__ == "__main__":
    dem1()
    dem2()