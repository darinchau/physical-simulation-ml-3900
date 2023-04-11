## This module is code from week one. We try to keep everything outside

import sys
import os
import shutil
import time
from load import load_data_week_1, make_anim_week_2, save_h5, peek_h5
import numpy as np
import matplotlib.pyplot as plt
from train import *
from tqdm import tqdm
import re
from multiprocessing import Process
from typing import Iterable

# Returns true if the pattern says the number of splits is ass
def too_many_split(e: ValueError):
    st = e.args[0]
    pattern = r"^Cannot have number of splits n_splits=[0-9]* greater than the number of samples: n_samples=[0-9]*.$"
    return bool(re.match(pattern, st))

# Create a folder and if folder exists, remove/overwrite everything inside :D
def create_folder_directory(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return folder_path
    else:
        i = 1
        while True:
            new_path = f"{folder_path} ({i})"
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                return new_path
            i += 1

# Get the first n inputs as inputs, data and train index
def get_first_n_inputs(n):
    inputs = np.arange(101)*0.75/100
    data = load_data_week_1()
    train_idx = list(range(n))
    return inputs, data, train_idx


# Takes in a regressor and trains the regressor on 1 - 101 samples
# to_test: An iterator of numbers for the "n" in first n data
def model_test(regressor: Regressor,
               to_test: Iterable[int],
               use_progress_bar = False,
               verbose = False):
    # Define history array for plotting
    hist = []
    hist_idxs = []

    # Create the path to save the datas
    model_name = regressor.model_name

    path = f"./Datas/Week 2/{model_name}"
    path = create_folder_directory(path)
    logs_file = f"{path}/{model_name} logs.txt"

    # Set some information on the regressor
    regressor.set_path(path)

    # Create the logs and description
    logs = []

    desc = f"Training {model_name}"
    desc += " " * (45 - len(desc))

    # Save all the predictions to generate gif later
    predictions = {}
    predictions["original"] = load_data_week_1()

    # Make the iterator depending on whether we need tqdm (progress bar)
    if use_progress_bar:
        # Length of progress bar
        bar_length = 20
        to_test = tqdm(to_test, desc = desc, bar_format=f"{{l_bar}}{{bar:{bar_length}}}{{r_bar}}{{bar:-{bar_length}b}}")

    # Branch off the save animation processes using multiprocessing because they take a long time
    save_anim_processes = []

    for n in to_test:
        # Set metadata on regressor
        regressor.set_input_name(f"first {n}")

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
        pred = regressor.predict(inputs.reshape((-1, 1))).reshape((101, 129, 17))

        # Calculate errors :) In order is: RMSE, worst over all data, RMSE for last 10
        rmse = np.sqrt(np.mean((data - pred)**2))
        worst = np.max(np.abs(data - pred))
        rmse_last_10 = np.sqrt(np.mean((data[-10:] - pred[-10:])**2))

        # Save the historical data to plot the graph
        hist.append((rmse, worst, rmse_last_10))
        hist_idxs.append(n)

        # Make and save the animation using multiprocessing
        p = Process(target=make_anim_week_2, args=(pred, data, rmse, worst, f"{path}/first_{n}.gif", f"{model_name} with first {n} data"))
        p.start()
        save_anim_processes.append(p)

        # Save all the predictions
        predictions[f"frame {n}"] = pred

        # Create the logs
        log = f"{model_name} using the first {n} data: RMSE = {rmse}, worst = {worst}"
        logs.append(log)

        # Print the logs if necessary
        if verbose:
            print(log)

    # Create the file and overwrite as blank if necessary
    with open(logs_file, "w", encoding="utf-8") as f:
        f.write(regressor.train_info)
        f.write("\n\n\n")

    # Append all the logs
    with open(logs_file, "a") as f:
        for log in logs:
            f.write(log)
            f.write("\n")

    # Save the predictions
    # THe predictions dict should be saved as this format:
    # frame 1
    #     data
    #         (101, 129, 17)
    # frame 2
    #     data
    #         (101, 129, 17)
    # frame 3
    #     data
    #         (101, 129, 17)
    # frame 4
    #     data
    #         (101, 129, 17)
    # ... ...
    # original
    #     data
    #         (101, 129, 17)
    # Open an HDF5 file in write mode
    save_h5(predictions, f"{path}/predictions.h5")

    # Plot everything using matplotlib
    plt.figure()
    plt.plot(hist_idxs, hist)
    plt.yscale("log")
    plt.legend(["RMSE", "Worst error", "RMSE for last 10"])
    plt.title(f"Result prediction using {model_name} from first n data")
    plt.savefig(f"{path}/Predicted {model_name}.png")

    # Join back the animation saving processes
    for process in save_anim_processes:
        process.join()

############################################
#### Helper Functions for model testing ####
############################################

# For parallel model testing
def execute_test(model, num_to_test):
    model_test(model, to_test=num_to_test, verbose=True, use_progress_bar=False)

# Sequentially/Parallelly(?) train all models and test the results
# calculates using the first num_to_test results
# if sequential, show progress bar if pbar
def test_all_models(models, sequential, to_test, pbar=False):
    t = time.time()

    if sequential:
        for model in models:
            model_test(model, to_test, verbose=False, use_progress_bar=pbar)
    else:
        processes: list[Process] = []
        for model in models:
            p = Process(target=execute_test, args=(model, to_test))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    print(f"Total time taken: {round(time.time() - t, 3)}")

def batch_1():
    test_all_models([
        DecisionTreeRegression(),
        RidgeCVRegression(),
        GaussianRegression(),
        SGDRegression(l1_ratio=0.01),
        SGDRegression(l1_ratio=0.15),
        SGDRegression(l1_ratio=0.5),
        SGDRegression(l1_ratio=0.85),
        SGDRegression(l1_ratio=0.99),
        PassiveAggressiveRegression(),
        PassiveAggressiveRegression(C = 0.1, max_iter=1000000),
    ], sequential = False, to_test = range(1, 91))

def batch_2():
    test_all_models([
        LinearRegression(),
        MultiTaskLassoCVRegression(),
        MultiTaskElasticNetCVRegression(l1_ratio=0.01),
        MultiTaskElasticNetCVRegression(l1_ratio=0.25),
        MultiTaskElasticNetCVRegression(l1_ratio=0.5),
        MultiTaskElasticNetCVRegression(l1_ratio=0.75),
        MultiTaskElasticNetCVRegression(l1_ratio=0.99),
        BayesianRidgeRegression(n_iter=300),
        BayesianRidgeRegression(n_iter=3000, tol = 0.0001),
    ], sequential = False, to_test = range(1, 91))


def batch_3():
    tt = (1, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 75, 90)
    test_all_models([
        # Week1Net1(epochs = (600, 250), show_training_logs=True),
        # Week2Net1(epochs = (600, 250), show_training_logs=True),
        Week2Net2(epochs = (600, 10), show_training_logs=True),
        # Week2Net3(epochs = (600, 250), show_training_logs=True),
    ], sequential=False, to_test=(1, 3), pbar=False)

if __name__ == "__main__":
    if sys.argv[1] == 'batch1':
        batch_1()
    elif sys.argv[1] == 'batch2':
        batch_2()
    elif sys.argv[1] == 'batch3':
        batch_3()
