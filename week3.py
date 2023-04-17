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
def get_first_n_inputs(n, use_e_density = False):
    inputs = (np.arange(101)*0.75/100).reshape((101, 1))
    if use_e_density:
        e_density = load_e_density().reshape((101, 2193))
        inputs = np.concatenate([inputs, e_density], axis = 1)
    data = load_elec_potential()
    train_idx = list(range(n))
    return inputs, data, train_idx

# Takes in a regressor and trains the regressor on 1 - 101 samples
# to_test: An iterator of numbers for the "n" in first n data
def model_test(regressor: Regressor,
               to_test: Iterable[int],
               use_progress_bar = False,
               verbose = False,
               use_e_density = False):
    # Create the path to save the datas
    model_name = regressor.model_name

    # Exit early if the model does not interop with electron density
    if use_e_density and not regressor.can_use_electron_density:
        return

    if use_e_density:
        path = f"{PATH_PREPEND}/{model_name} edensity"
    else:
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
        inputs, data, train_idx = get_first_n_inputs(n, use_e_density)

        num_features = inputs.shape[1]

        # Fit the model
        try:
            regressor.fit_model(inputs, data, train_idx, skip_error=True)
        except RegressorFitError as e:
            continue

        # Calculate the model and compare with actual data
        pred = regressor.predict(inputs.reshape((-1, num_features))).reshape((101, 129, 17))

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
        if use_e_density:
            f.write("\n\n Trained with electron density")
        else:
            f.write("\n\n Trained without electron density")

    # Make some animations
    make_plots(path, model_name)

############################################
#### Helper Functions for model testing ####
############################################

# For parallel model testing
def execute_test(model, num_to_test, use_e_density):
    model_test(model, to_test=num_to_test, verbose=True, use_progress_bar=False, use_e_density=use_e_density)

# Sequentially/Parallelly(?) train all models and test the results
# calculates using the first num_to_test results
# if sequential, show progress bar if pbar
def test_all_models(models, sequential, to_test, pbar=False, use_e_density = False):
    t = time.time()

    if sequential:
        for model in models:
            model_test(model, to_test, verbose=False, use_progress_bar=pbar, use_e_density=use_e_density)
    else:
        processes: list[Process] = []
        for model in models:
            p = Process(target=execute_test, args=(model, to_test, use_e_density))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    print(f"Total time taken: {round(time.time() - t, 3)}")

# The values of n to test
GOAL_TEST = (1, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 75, 90)

if __name__ == "__main__":
    # test_all_models([
    #     RidgeCVRegression(),
    #     GaussianRegression(),
    #     LinearRegression(),
    #     SGDRegression(),
    #     MultiTaskLassoCVRegression(),
    #     MultiTaskElasticNetCVRegression(),
    #     BayesianRidgeRegression(),
    #     # GLH1Regression(),
    #     # GLH2Regression(),
    #     # GLH3Regression(),
    #     # GLH4Regression(),
    #     # SimpleNetRegression()
    # ], sequential=True, to_test=GOAL_TEST, use_e_density=True)

    test_all_models([
        TCEPRegression(20)
    ], sequential=True, to_test=GOAL_TEST, use_e_density=False)
