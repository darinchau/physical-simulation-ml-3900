### This module contains all save/load/os related methods for our project ###

import os
import numpy as np
from numpy.typing import NDArray
import h5py
import torch
from torch import Tensor

SPACING_X = np.load("./simulation data/mesh_data_x.npy")
SPACING_Y = np.load("./simulation data/mesh_data_y.npy")
ELECTRIC_POTENTIAL = np.load("./simulation data/mesh_data_electrostatic_potential.npy")
ELECTRON_DENSITY = np.load("./simulation data/mesh_data_edensity.npy")
SPACE_CHARGE = np.load("./simulation data/mesh_data_space_charge.npy")
MATERIALS = np.load("./simulation data/mesh_data_materials.npy")
CONTACTS = np.load("./simulation data/mesh_data_contacts.npy")

# Get Cuda if cuda is available
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # return torch.device('cpu')

def load_vgs():
    return torch.arange(101).reshape(101, 1) / 100 * 0.75

# Load and return data. We expect data to be some 3 dimensional np array (N, rows, cols).
def load_elec_potential() -> Tensor:
    return torch.tensor(ELECTRIC_POTENTIAL)

def load_e_density() -> Tensor:
    return torch.tensor(np.nan_to_num(ELECTRON_DENSITY, nan=0))

def load_spacing() -> Tensor:
    return torch.tensor(SPACING_X), torch.tensor(SPACING_Y)

def load_space_charge() -> Tensor:
    return torch.tensor(np.nan_to_num(SPACE_CHARGE, nan=0))

def load_materials() -> Tensor:
    return torch.tensor(MATERIALS == 1)

def load_contacts() -> Tensor:
    return torch.tensor(CONTACTS)

### Save h5 files
def save_h5(d: dict[str, NDArray], path: str):
    # Open an HDF5 file in write mode
    with h5py.File(path, 'a') as f:
        # Loop through dictionary keys and add them as groups to the file
        for key in d.keys():
            # If key already exists then keep the original data
            if key in f:
                continue
            group = f.create_group(key)
            dataset = group.create_dataset('data', data=d[key], compression="gzip", compression_opts=9)

# Helper function to load and print the structure of a h5
def peek_h5(path: str):
    with h5py.File(path, 'r') as f:
        # Loop through keys of the file and print them
        for key in f.keys():
            print(key)
            group = f[key]

            # Loop through keys of the group and print them
            for subkey in group.keys():
                print('\t', subkey)
                dataset = group[subkey]

                # Print the shape and first element of the dataset
                print('\t\t', dataset.shape)


# Deletes everythin in a folder
def delete_folder_contents(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# Create a folder directory, and compare the logs - if we found a training with identical hyperparameters, append to that
# root = "./Datas/Week 6"
def get_folder_directory(root: str, model):
    folder_path = f"{root}/{model.name}"
    return get_folder_directory_recursive(folder_path, model)

def get_folder_directory_recursive(folder_path: str, model):
    # If the module does not exist at all :)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return folder_path
    
    # If the folder exists but the logs file does not, this means it is probably the remnants of the previous training
    # where some sort of error probably occured. Use this folder
    if not os.path.isfile(f"{folder_path}/logs.txt"):
        print("Could not find folder log contents - deleting file contents")
        delete_folder_contents(folder_path)
        return folder_path
    
    # Find the logs file. If the logs have the same content we would have generated, then return the existing file path
    with open(f"{folder_path}/logs.txt", 'r') as f:
        st = f.read()
    if st.strip() == model.logs.strip():
        return folder_path
    
    # Create something like Model (1), Model (2) and so on via recursion
    if '(' not in folder_path.split("/")[-1]:
        return get_folder_directory_recursive(f"{folder_path} (1)", model)
    
    num = int(folder_path.split("(")[1][:-1])
    path = folder_path.split("(")[0][:-1]
    return get_folder_directory_recursive(f"{path} ({num + 1})", model)
