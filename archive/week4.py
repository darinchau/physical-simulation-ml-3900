## This module is code from week one. We try to keep everything outside

import sys
import os
from os import path
import shutil
import time
from load import *
import numpy as np
import matplotlib.pyplot as plt
from archive.train import *
from tqdm import tqdm
from multiprocessing import Process
from typing import Iterable
from dataclasses import dataclass

# The place to store all your datas
PATH_PREPEND = "./Datas/Week 4"

# Get the first n inputs as inputs, data and train index
def get_data(use_e_density: bool, use_space_charge: bool):
    inputs = (np.arange(101)*0.75/100).reshape((101, 1))
    
    if use_e_density:
        e_density = load_e_density().reshape((101, 2193))
        e_density /= np.max(e_density)
        inputs = np.concatenate([inputs, e_density], axis = 1)
        
    if use_space_charge:
        space_charge = load_space_charge().reshape((101, 2193))
        space_charge /= np.max(np.abs(space_charge))
        inputs = np.concatenate([inputs, space_charge], axis = 1)
    
    data = load_elec_potential()
    return inputs, data

@dataclass
class TrainingIndex:
    name: str
    indices: list[int]
    
    def __iter__(self):
        return iter(self.indices)
    
# Given the regressors return the log:
def get_logs(model_name: str, regressor: Regressor):
    return f"{model_name}\n\n{regressor.train_info}"

# Create a folder directory, and compare the logs - if we found a training with identical hyperparameters, append to that
def create_folder_directory(folder_path, model_name: str, regressor: Regressor):
    # If the module does not exist at all :)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return folder_path
    
    # Find the logs file. If the logs have the same content we would have generated, then return the existing file path
    if os.path.isfile(f"{folder_path}/logs.txt"):
        with open(f"{folder_path}/logs.txt", 'r') as f:
            st = f.read()
        logs = get_logs(model_name, regressor)
        if st.strip() == logs.strip():
            return folder_path
    
    # Create something like Model (1), Model (2) and so on
    if '(' not in folder_path.split("/")[-1]:
        return create_folder_directory(f"{folder_path} (1)", model_name, regressor)
    
    num = int(folder_path.split("(")[1][:-1])
    path = folder_path.split("(")[0][:-1]
    return create_folder_directory(f"{path} ({num + 1})", model_name, regressor)

# Takes in a regressor and trains the regressor on 1 - 101 samples
# to_test: An iterator of numbers for the "n" in first n data
def train_model(regressor: Regressor,
               to_test: Iterable[TrainingIndex],
               model_name: str,
               use_e_density = False,
               use_space_charge = False):
    # Create the file and logs
    path = create_folder_directory(f"{PATH_PREPEND}/{model_name}", model_name, regressor)
    logs_file = f"{path}/logs.txt"

    # Set some information on the regressor
    regressor.set_path(path)
    
    # Save all the predictions to generate gif later
    predictions = {}

    for idx in to_test:
        # Preprocess the data
        inputs, data = get_data(use_e_density, use_space_charge)
        
        # Get the training idx
        train_idx = list(idx)

        # Get number of features
        num_features = inputs.shape[1]

        # Fit the model
        try:
            regressor.fit_model(inputs, data, train_idx, skip_error=True)
        except RegressorFitError as e:
            continue

        # Calculate the model and compare with actual data
        pred = regressor.predict(inputs.reshape((-1, num_features))).reshape((101, 129, 17))

        # Save all the predictions
        predictions[idx.name] = pred

        # Print some indication on finishing training
        print(f"Done {model_name} using {idx.name[1:]}" if idx.name[0] == "_" else f"Done {model_name} using {idx.name}")

        # Save the history at each step
        save_h5(predictions, f"{path}/predictions.h5")

    # Create the file and overwrite as blank if necessary
    # Write the training info of the regressor (mostly hyperparameters)
    with open(logs_file, "w", encoding="utf-8") as f:
        f.write(get_logs(model_name, regressor))

# Helper function to test the model both with and without electron density
# This is to interop with space charge stuff in the future
def models_test(regressor: Regressor, to_test: Iterable[TrainingIndex]):
    # Create the path to save the datas
    model_name = regressor.model_name
    
    train_model(regressor, to_test, model_name, use_e_density = False)
    
    if regressor.max_num_features >= 2194:
        train_model(regressor, to_test, f"{model_name}-d", use_e_density = True)
        train_model(regressor, to_test, f"{model_name}-sc", use_space_charge = True)
    
    if regressor.max_num_features >= 4386:
        train_model(regressor, to_test, f"{model_name}-scd", use_space_charge = True, use_e_density = True)
        
############################################
#### Helper Functions for model testing ####
############################################

# Wrapper function for parallel model testing
def execute_test(model, to_test):
    models_test(model, to_test)

# Sequentially/Parallelly(?) train all models and test the results
# calculates using the first num_to_test results
# if sequential, show progress bar if pbar
def test_all_models(models: list[Regressor], sequential: bool, to_test: Iterable[TrainingIndex]):
    t = time.time()

    if sequential:
        for model in models:
            models_test(model, to_test)
    else:
        processes: list[Process] = []
        for model in models:
            p = Process(target=execute_test, args=(model, to_test))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    print(f"Total time taken: {round(time.time() - t, 3)}")

def training_1():
    test_all_models([
        LinearRegression(),
        RidgeCVRegression(),
        GaussianRegression(),
        GLH1Regression(),
        LLH3Regression(),
        LLH4Regression(),
        LLH5Regression()
    ], to_test = [
        TrainingIndex("First 5", range(5)),
        TrainingIndex("First 20", range(20)),
        TrainingIndex("First 30", range(30)),
        TrainingIndex("First 40", range(40)),
        TrainingIndex("First 60", range(60)),
        TrainingIndex("First 75", range(75)),
        TrainingIndex("First 90", range(90)),
        TrainingIndex("15 to 45", range(15, 45)),
        TrainingIndex("20 to 40", range(20, 40)),
        TrainingIndex("40 to 60", range(40, 60)),
        TrainingIndex("25 to 35", range(25, 35)),
        TrainingIndex("20 to 50", range(20, 50)),
        TrainingIndex("30 to 50", range(30, 50)),
        TrainingIndex("29 and 30 and 31", [29, 30, 31]),
    ], sequential = True)

# Shows the distribution of the log of the data
def show_log_distribution(data, data_name: str):
    data = np.array(data)
    data = np.log10(data[data > 0])
    min_log = int(np.min(data)) - 1
    max_log = int(np.max(data)) + 1
    
    x, y = [], []
    for i in range(min_log, max_log):
        count = ((i < data) & (data < i+1)).sum()
        x.append(i)
        y.append(count)
    
    fig, ax = plt.subplots()
    ax.bar(x, y)
    fig.suptitle(f"Log distribution of {data_name}")
    fig.savefig(f"{data_name}.png")
    
    try:
        ax.set_yscale('log')
        fig.savefig(f"{data_name} log.png")
    except ValueError:
        pass

def plot_data():
    anim = AnimationMaker()
    
    anim.add_data(load_elec_potential(), "Electric potential")
    anim.add_data(load_e_density(), "Electron density", vmin = 1e13)
    anim.add_data(load_log_e_density(), "Electron density (log(x))", vmin = 13)
    anim.add_data(load_space_charge(), "Space charge")
    
    lsc = np.abs(load_space_charge())
    lsc[lsc == 0] = 1
    lsc = np.log10(lsc)
    anim.add_data(lsc, "Space charge (log|x|)", vmin = 13)
    
    anim.add_text([f"Frame {i} - {i * 0.0075:.4f}V" for i in range(101)])
    anim.save("data.gif")
    
def plot_data_2(model_name):
    d = DataVisualizer()
    with h5py.File(f"{PATH_PREPEND}/{model_name}/predictions.h5", 'r') as f:
        for key in ("First 5", "First 20", "20 to 40", "40 to 60"):
            d.add_data(f[key]["data"][:], key)
    d.add_data(load_elec_potential(), "Original", thickness=3)
    d.show()

if __name__ == "__main__":
    plot_data_2("Linear regression-scd")
