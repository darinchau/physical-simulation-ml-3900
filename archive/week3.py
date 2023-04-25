## This module is code from week one. We try to keep everything outside

import sys
import os
import shutil
import time
from load import *
import numpy as np
import matplotlib.pyplot as plt
from archive.train import *
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

NO_E_DENSITY = 0
USE_E_DENSITY = 1
USE_NORMALIZED_E_DENSITY = 2

# Get the first n inputs as inputs, data and train index
def get_first_n_inputs(n, use_e_density = False):
    inputs = (np.arange(101)*0.75/100).reshape((101, 1))
    if use_e_density:
        e_density = load_e_density().reshape((101, 2193))
        e_density /= np.max(e_density)
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
               use_e_density = NO_E_DENSITY):
    # Create the path to save the datas
    model_name = regressor.model_name

    # Exit early if the model does not interop with electron density
    if not (use_e_density == NO_E_DENSITY or regressor.max_num_features):
        return

    if use_e_density == USE_E_DENSITY:
        path = f"{PATH_PREPEND}/{model_name} edensity"
    elif use_e_density == NO_E_DENSITY:
        path = f"{PATH_PREPEND}/{model_name}"
    else:
        path = f"{PATH_PREPEND}/{model_name} normed"

    if use_e_density == USE_E_DENSITY:
        model_name = f"{model_name} with electron density"
    elif use_e_density == USE_NORMALIZED_E_DENSITY:
        model_name = f"{model_name} with normalized electron density"

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
        f.write(model_name)
        f.write("\n\n")
        f.write(regressor.train_info)

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
def test_all_models(models, sequential, to_test, pbar=False, use_e_density = NO_E_DENSITY):
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

def task1():
    test_all_models([
        RidgeCVRegression(),
        GaussianRegression(),
        LinearRegression(),
    ], sequential=False, to_test=GOAL_TEST, use_e_density=USE_E_DENSITY)

def task2():
    test_all_models([
        RidgeCVRegression(),
        GaussianRegression(),
        LinearRegression(),
        GLH1Regression(),
        GLH2Regression(),
        GLH3Regression(),
        GLH4Regression(),
    ], sequential=False, to_test=GOAL_TEST, use_e_density=USE_NORMALIZED_E_DENSITY)

def task3():
    test_all_models([
        TCEPRegression(5),
        TCEPRegression(20),
        TCEPRegression(5, num_epochs=50000),
        TCEPRegression(20, num_epochs=50000),
    ], sequential=True, to_test=GOAL_TEST, use_e_density=NO_E_DENSITY)

def task4():
    test_all_models([
        PolynomialRegression(1),
        PolynomialRegression(2),
        PolynomialRegression(3),
        PolynomialRegression(4),
        PolynomialRegression(5),
        PolynomialRegression(6),
        PolynomialRegression(7),
        PolynomialRegression(8),
    ], sequential=False, to_test=GOAL_TEST, use_e_density=NO_E_DENSITY)

def task5():
    test_all_models([
        PolynomialRegression(2),
    ], sequential=True, to_test=GOAL_TEST, use_e_density=USE_E_DENSITY)

def task6():
    test_all_models([
        PolynomialRegression(2),
    ], sequential=True, to_test=GOAL_TEST, use_e_density=USE_NORMALIZED_E_DENSITY)

def task7():
    test_all_models([
        MultiTaskLassoCVRegression(),
        MultiTaskElasticNetCVRegression(),
        BayesianRidgeRegression(),
        SGDRegression(),
    ], sequential=True, to_test=GOAL_TEST, use_e_density=USE_NORMALIZED_E_DENSITY)

def task8():
    test_all_models([
        RidgeCVRegression(),
        GaussianRegression(),
        LinearRegression(),
    ], sequential=False, to_test=GOAL_TEST, use_e_density=NO_E_DENSITY)

def task9():
    test_all_models([
        TCEPRegression(5, num_epochs=10000, tcep=TCEP2Net),
    ], sequential=True, to_test=(5, 10, 20, 40, 60, 90), use_e_density=NO_E_DENSITY)

def make_animation():
    anim = AnimationMaker()
    anim.add_data(load_elec_potential(), "Electric potential")
    anim.add_data(load_e_density(), "Electron density")

    log_e_density = load_e_density()
    log_e_density[log_e_density < 1] = 1
    log_e_density = np.log10(log_e_density)

    anim.add_data(log_e_density, "Electron density log", vmin = 0)

    anim.add_text([f"Frame {i}, {round(0.0075*i, 4)}V" for i in range(101)])

    anim.save(path="./data.gif", suptitle="Data plots")


if __name__ == "__main__":
    # task1()
    # task2()
    # task3()
    # task4()
    # task5()
    # task6()
    # task7()
    # task8()
    # task9()

    make_animation()
