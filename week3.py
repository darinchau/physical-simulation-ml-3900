## This module is code from week one. We try to keep everything outside

import sys
import os
import shutil
import time
from load import *
import numpy as np
import matplotlib.pyplot as plt
from train import *
from tqdm import tqdm
from multiprocessing import Process
from typing import Iterable
import h5py

# The place to store all your datas
PATH_PREPEND = "./Datas/Week 3"

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
    data = load_elec_potential()
    train_idx = list(range(n))
    return inputs, data, train_idx

# Takes in a regressor and trains the regressor on 1 - 101 samples
# to_test: An iterator of numbers for the "n" in first n data
def model_test(regressor: Regressor,
               to_test: Iterable[int],
               use_progress_bar = False,
               verbose = False):
    # Create the path to save the datas
    model_name = regressor.model_name

    path = f"{PATH_PREPEND}/{model_name}"
    path = create_folder_directory(path)
    logs_file = f"{path}/{model_name} logs.txt"

    # Set some information on the regressor
    regressor.set_path(path)

    desc = f"Training {model_name}"
    desc += " " * (45 - len(desc))

    # Save all the predictions to generate gif later
    predictions = {}

    # Make the iterator depending on whether we need tqdm (progress bar)
    if use_progress_bar:
        # Length of progress bar
        bar_length = 20
        to_test = tqdm(to_test, desc = desc, bar_format=f"{{l_bar}}{{bar:{bar_length}}}{{r_bar}}{{bar:-{bar_length}b}}")

    for n in to_test:
        # Set metadata on regressor
        regressor.set_input_name(f"first {n}")

        # Preprocess the data
        inputs, data, train_idx = get_first_n_inputs(n)

        # Fit the model
        try:
            regressor.fit(inputs, data, train_idx, skip_error=True)
        except RegressorFitError as e:
            continue

        # Calculate the model and compare with actual data
        pred = regressor.predict(inputs.reshape((-1, 1))).reshape((101, 129, 17))

        # Save all the predictions
        predictions[f"frame {n}"] = pred

        # Print some indication on finishing first n
        log = f"Done {model_name} using the first {n} data"
        print(log)

        # Save the history at each step
        save_h5(predictions, f"{path}/predictions.h5")

    # Create the file and overwrite as blank if necessary
    # Write the training info of the regressor (mostly hyperparameters)
    with open(logs_file, "w", encoding="utf-8") as f:
        f.write(regressor.train_info)

    # Make some animations
    make_plots(path, model_name)

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

# The values of n to test
GOAL_TEST = (1, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 75, 90)

if __name__ == "__main__":
    test_all_models([
        # RidgeCVRegression(),
        # GaussianRegression(),
        # LinearRegression(),

        # GLH1Regression((30, 69, 30)),
        # GLH1Regression((35, 59, 35)),
        # GLH1Regression((40, 49, 40)),
        # GLH1Regression((45, 39, 45)),

        # GLH2Regression((30, 69, 30)),
        # GLH2Regression((35, 59, 35)),
        # GLH2Regression((40, 49, 40)),
        # GLH2Regression((45, 39, 45)),

        # GLH3Regression((30, 69, 30)),
        # GLH3Regression((35, 59, 35)),
        # GLH3Regression((40, 49, 40)),
        # GLH3Regression((45, 39, 45)),
    ], sequential=False, to_test=GOAL_TEST)
