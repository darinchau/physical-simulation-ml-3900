## This module is code from week one. We try to keep everything outside

import os
import shutil
import time
from train import train_mlp, train_gaussian_process, lazy_predict
from load import load_data_week_1, wrap_data, make_anim, index_exclude, index_include, make_anim_week_2
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from raw_train import Regressor, DecisionTreeRegression, RidgeCVRegression, GaussianRegression, SGDRegression, PassiveAggressiveRegression, LinearRegression, MultiTaskLassoCVRegression, MultiTaskElasticNetCVRegression, BayesianRidgeRegressor
from tqdm import trange
import re
import multiprocessing

# Returns true if the pattern says the number of splits is ass
def too_many_split(e: ValueError):
    st = e.args[0]
    pattern = r"^Cannot have number of splits n_splits=[0-9]* greater than the number of samples: n_samples=[0-9]*.$"
    return bool(re.match(pattern, st))

# Create a folder and if folder exists, remove/overwrite everything inside :D
def create_folder_directory(folder_path):
    if os.path.exists(folder_path):
        try:
            # Delete the folder and its contents
            shutil.rmtree(folder_path)
        except:
            raise Exception(f"Error deleting folder: {folder_path}")
    try:
        # Create the folder directory
        os.makedirs(folder_path)
    except:
        raise Exception(f"Error creating folder: {folder_path}")

# Get the first n inputs as inputs, data and train index
def get_first_n_inputs(n):
    inputs = np.arange(101)*0.75/100
    data = load_data_week_1()
    train_idx = list(range(n))
    return inputs, data, train_idx


# Takes in a regressor and trains the regressor on 1 - 101 samples
def model_test(regressor: Regressor):
    hist = []
    hist_idxs = []

    # Create the path to save the datas
    model_name = regressor.model_name

    path = f"./Datas/Week 2/{model_name}"
    create_folder_directory(path)
    logs_file = f"{path}/{model_name} logs.txt"

    logs = []

    desc = f"Training {model_name}"
    desc += " " * (45 - len(desc))

    for n in trange(1, 101, desc = desc, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        # Preprocess the data
        inputs, data, train_idx = get_first_n_inputs(n)

        # Fit the model
        try:
            regressor.fit(inputs, data, train_idx, skip_error=True)
        except ValueError as e:
            if too_many_split(e) is True:
                continue
            raise e

        # Calculate the model and compare with actual data
        idxs = inputs.reshape((-1, 1))
        pred = regressor.predict(idxs)
        pred = pred.reshape((101, 129, 17))

        # Make and save the animation. This calculates and returns the errors during the process
        rmse, worst = make_anim_week_2(pred, data, f"{path}/first_{n}.gif", prediction_name = f"{model_name} with first {n} data")

        # Create the logs
        log = f"{model_name} using the first {n} data: RMSE = {rmse}, worst = {worst}"
        logs.append(log)

        # Plot the graph
        hist.append((rmse, worst))
        hist_idxs.append(n)

    # Create the file and overwrite as blank if necessary
    with open(logs_file, 'w') as f:
        f.write('')

    with open(logs_file, 'a') as f:
        for log in logs:
            f.write(log)
            f.write("\n")

    # Plot everything
    plt.figure()
    plt.plot(hist_idxs, hist)
    plt.yscale('log')
    plt.legend(['RMSE', 'Worst error'])
    plt.title(f"Result prediction using {model_name} from first n data")
    plt.savefig(f"{path}/Predicted {model_name}.png")

def test_anim():
    _, predicted_data, _ = get_first_n_inputs(10)
    pred = predicted_data + np.random.random(predicted_data.shape) * 0.01
    make_anim_week_2(pred, predicted_data, "hiya.gif", "hehehaha predictor with first -1 data")

if __name__ == "__main__":
    t = time.time()

    # Define a list of model instances to test
    models = [
        DecisionTreeRegression(),
        RidgeCVRegression(),
        GaussianRegression(),
        SGDRegression(),
        PassiveAggressiveRegression(),
        LinearRegression(),
        MultiTaskLassoCVRegression(),
        MultiTaskElasticNetCVRegression(),
        BayesianRidgeRegressor()
    ]

    # # Define a function to execute the model_test function in parallel
    # def parallel_execution(model):
    #     model_test(model)

    # # Create a multiprocessing pool with the number of cores available
    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # # Use the pool to execute the function in parallel for each model
    # results = pool.map(parallel_execution, models)

    # # Close the pool to free up resources
    # pool.close()

    for model in models:
        model_test(model)

    print(f"Total time taken: {round(time.time() - t, 3)}")
