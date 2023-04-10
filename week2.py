## This module is code from week one. We try to keep everything outside

import os
import shutil
import time
from load import load_data_week_1, make_anim_week_2
import numpy as np
import matplotlib.pyplot as plt
from train import *
from tqdm import trange
import re
from multiprocessing import Process

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
def model_test(regressor: Regressor,
               num_to_test,
               use_progress_bar = False,
               verbose = False,
               save_prediction_as_anim=True):
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

    # Save all the predictions instead of saving the gifs to save space
    if not save_prediction_as_anim:
        predictions = np.zeros((1 + num_to_test, 101, 129, 17))
        predictions[0] = load_data_week_1()

    # Make the iterator depending on whether we need tqdm (progress bar)
    iterator = trange(1, num_to_test, desc = desc, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') if use_progress_bar else range(1, num_to_test)

    # Branch off the save animation processes because they take a long time
    if save_prediction_as_anim:
        save_anim_processes = []

    for n in iterator:
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

        # Root mean square error and worst absolute error
        rmse = np.sqrt(np.mean((data - pred)**2))
        worst = np.max(np.abs(data - pred))

        if save_prediction_as_anim:
            # Make and save the animation. This calculates and returns the errors during the process
            p = Process(target=make_anim_week_2, args=(pred, data, rmse, worst, f"{path}/first_{n}.gif", f"{model_name} with first {n} data"))
            p.start()
            save_anim_processes.append(p)
        else:
            # Save all the predictions instead of the gifs to save space
            predictions[n] = pred

        # Create the logs
        log = f"{model_name} using the first {n} data: RMSE = {rmse}, worst = {worst}"
        logs.append(log)

        if verbose:
            print(log)

        # Plot the graph
        hist.append((rmse, worst))
        hist_idxs.append(n)

    # Create the file and overwrite as blank if necessary
    with open(logs_file, 'w', encoding="utf-8") as f:
        f.write(regressor.train_info)
        f.write("\n\n\n")

    # Append all the logs
    with open(logs_file, 'a') as f:
        for log in logs:
            f.write(log)
            f.write("\n")

    # Save the predictions array
    if not save_prediction_as_anim:
        np.save(f"{path}/predictions.npy", predictions)

    # Plot everything
    plt.figure()
    plt.plot(hist_idxs, hist)
    plt.yscale('log')
    plt.legend(['RMSE', 'Worst error'])
    plt.title(f"Result prediction using {model_name} from first n data")
    plt.savefig(f"{path}/Predicted {model_name}.png")

    if save_prediction_as_anim:
        for process in save_anim_processes:
            process.join()

############################################
#### Helper Functions for model testing ####
############################################

# For parallel model testing
def execute_test(model, num_to_test):
    model_test(model, num_to_test=num_to_test, verbose=True, use_progress_bar=False)

# Sequentially/Parallelly(?) train all models and test the results
def test_all_models(models, sequential, num_to_test=101):
    t = time.time()

    if sequential:
        for model in models:
            model_test(model, num_to_test=num_to_test, verbose=False, use_progress_bar=True)
    else:
        processes: list[Process] = []
        for model in models:
            p = Process(target=execute_test, args=(model, num_to_test))
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
        LinearRegression(),
        MultiTaskLassoCVRegression(),
        MultiTaskElasticNetCVRegression(l1_ratio=0.01),
        MultiTaskElasticNetCVRegression(l1_ratio=0.25),
        MultiTaskElasticNetCVRegression(l1_ratio=0.5),
        MultiTaskElasticNetCVRegression(l1_ratio=0.75),
        MultiTaskElasticNetCVRegression(l1_ratio=0.99),
        BayesianRidgeRegression(n_iter=300),
        BayesianRidgeRegression(n_iter=3000, tol = 0.0001),
    ], sequential = False)

def test_save_load():
    test_all_models([
        Week1Net1(epochs=30)
    ], sequential=True, num_to_test=2)

    model = Week1Net1()
    model.load("./Datas/Week 2/Week 1 Net 1/first 1.pt")
    inputs = np.arange(101)*0.75/100
    model.predict(inputs.reshape((-1, 1))).reshape((101, 129, 17))

if __name__ == "__main__":
    test_all_models([
        LinearRegression()
    ], sequential=True, num_to_test=3)
