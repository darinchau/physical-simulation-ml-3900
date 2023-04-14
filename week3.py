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
import re
from multiprocessing import Process
from typing import Iterable
import h5py

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
    data = load_elec_potential()
    train_idx = list(range(n))
    return inputs, data, train_idx

#Utility function for log to handle log 0
def log_array(arr):
    arr_log10 = np.log10(arr)
    arr_log10[arr == 0] = np.min(arr_log10[arr != 0]) - 1
    return arr_log10

def make_anim(path, model_name):
    with h5py.File(f"{path}/predictions.h5", 'r') as f:
        # Keep note of the frame errors
        frame_errors = {k: [] for k in f.keys()}

        # Loop through keys of the file and print them
        original_data = load_elec_potential()

        for key in f.keys():
            # Prediction, the index is to change it to numpy array
            pred = f[key]['data'][:]

            # First plot is the animation
            # Animation :D
            anim = AnimationMaker(101)
            anim.add_plot(original_data, "original")
            anim.add_plot(pred, "prediction", (np.min(original_data), np.max(original_data)))
            anim.add_plot(np.abs(pred - original_data), "error", (0, None))
            anim.add_plot(log_array(np.abs(pred - original_data)), "error (log10)")

            # Total errors (RMSE, Worst errors, Worst error for the last 10 frames)
            rmse = np.sqrt(np.mean((pred - original_data) ** 2))
            worst = np.max(np.abs(pred - original_data))
            rmse_last_10_frames = np.sqrt(np.mean((pred[-10:] - original_data[-10:]) ** 2))
            worst_last_10_frames = np.max(np.abs(pred[-10:] - original_data[-10:]))

            anim.add_text(f"RMSE: {round(rmse, 5)}, worst = {round(worst, 5)}")
            anim.add_text(f"RMSE(last 10 frames): {round(rmse_last_10_frames, 5)}, worst(last 10 frames) = {round(worst_last_10_frames, 5)}")
            anim.add_text([f"Frame {i}: {0.0075 * i} V" for i in range(101)])

            anim.plot(f"Results from {model_name} with first {key[6:]} data", f"{path}/first {key[6:]}.gif")

            # Second plot is error each frame for different ns
            # Calculate RMSE for each frame
            # Uses a for loop to save memory. I know einsum is a thing but I dont know how to use it
            for i in range(101):
                rmse = np.sqrt(np.mean((pred[i] - original_data[i]) ** 2))
                frame_errors[key].append(rmse)

        # Plot error each frame
        fig, ax = plt.subplots()

        for key, value in frame_errors.items():
            # The indexing is on keys which is of the format "frame 123"
            # So all it does is to crop away the prepedn
            ax.plot(value, label=f"First {key[6:]}")

        # add legend to the plot
        ax.legend()

        # Title
        fig.suptitle("RMSE Error using the first n data across frames")

        # Show the thing
        fig.savefig(f"{path}/frame error.png")


# Takes in a regressor and trains the regressor on 1 - 101 samples
# to_test: An iterator of numbers for the "n" in first n data
def model_test(regressor: Regressor,
               to_test: Iterable[int],
               use_progress_bar = False,
               verbose = False):
    # Create the path to save the datas
    model_name = regressor.model_name

    path = f"./Datas/Week 3/{model_name}"
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
        except ValueError as e:
            if too_many_split(e) is True:
                continue
            raise e

        # Calculate the model and compare with actual data
        pred = regressor.predict(inputs.reshape((-1, 1))).reshape((101, 129, 17))

        # Save all the predictions
        predictions[f"frame {n}"] = pred

        # Create the logs
        log = f"Done {model_name} using the first {n} data"
        logs.append(log)
        print(log)

        # Save the history at each step
        save_h5(predictions, f"{path}/predictions.h5")

    # Create the file and overwrite as blank if necessary
    with open(logs_file, "w", encoding="utf-8") as f:
        f.write(regressor.train_info)
        f.write("\n\n\n")

    # Append all the logs
    with open(logs_file, "a") as f:
        for log in logs:
            f.write(log)
            f.write("\n")

    # Make some animations
    make_anim(path, model_name)

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

GOAL_TEST = (1, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 75, 90)

if __name__ == "__main__":
    test_all_models([
        GaussianLinearRegression(),
    ], sequential=True, to_test=GOAL_TEST)

    # make_anim("./Datas/Week 3/Gaussian Linear Hybrid 1", "GLH1")
