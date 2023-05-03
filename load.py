### This module contains all save/load/os related methods for our project ###

import os
import numpy as np
from numpy.typing import NDArray
import h5py
from models_base import Model
import torch
from torch import Tensor

SPACING_X = np.load("mesh_data_x.npy")
SPACING_Y = np.load("mesh_data_y.npy")
ELECTRIC_POTENTIAL = np.load("mesh_data_electrostatic_potential.npy")
ELECTRON_DENSITY = np.load("mesh_data_edensity.npy")
SPACE_CHARGE = np.load("mesh_data_space_charge.npy")

# Load and return data. We expect data to be some 3 dimensional np array (N, rows, cols).
def load_elec_potential() -> Tensor:
    return torch.tensor(ELECTRIC_POTENTIAL)

def load_e_density() -> Tensor:
    return torch.tensor(np.nan_to_num(ELECTRON_DENSITY, nan=0))

def load_spacing() -> Tensor:
    return torch.tensor(SPACING_X), torch.tensor(SPACING_Y)

def load_space_charge() -> Tensor:
    return torch.tensor(np.nan_to_num(SPACE_CHARGE, nan=0))

Q = 1.60217663e-19

def derivative_one_frame(data, x, y):
    # Initialize 
    frame_result = torch.zeros_like(data)

    # Ignores calculating at edges of the array
    for i in range(1, len(data) - 2):
        for j in range(1, len(data[0]) - 1):
            # unit of "eps_*": F/cm = C/(V*cm) 
            # unit of x and y are in um (converted to cm later)
            # unit of electrostatic potential is in V
            xr, yr, epr = x[i], y[j + 1], data[i][j + 1]
            xd, yd, epd = x[i + 1], y[j], data[i + 1][j]
            xc, yc, epc = x[i], y[j], data[i][j]
            xl, yl, epl = x[i], y[j - 1], data[i][j - 1]
            xu, yu, epu = x[i - 1], y[j], data[i - 1][j]
            
            # Currently assumed that only silicon is everywhere
            # Later, will need to adjust for the silicon-oxide interface
            relative_permittivity_silicon = 11.7
            
            # Convert free space permittivity to F/cm
            e0_cm = (8.85418782e-12) / 100
            
            # Actual dielectric permittivity = permittivity of free space * permittivity of material
            eps_l = (e0_cm * relative_permittivity_silicon)
            eps_r = (e0_cm * relative_permittivity_silicon)
            eps_u = (e0_cm * relative_permittivity_silicon)
            eps_d = (e0_cm * relative_permittivity_silicon)

            dx_right = eps_r * (epr - epc) / (yr - yc)
            dx_left = eps_l * (epc - epl) / (yc - yl)
            dxx = 2 * (dx_right - dx_left) / (yr - yl)

            dy_up = eps_u * (epu - epc) / (xu - xc)
            dy_down = eps_d * (epc - epd) / (xc - xd)
            dyy = 2 * (dy_up - dy_down) / (xu - xd)
            
            # the final unit of "ds" is in C/(cm * um^2). Multiply by 1e8 to convert the um^2 to cm^2 to get to C/(cm^3).
            div_cm = (dxx + dyy) * 1e8
            
            # Divide by constant Q (charge of electron)
            space_charge = div_cm / -1.60217663e-19
            
            # Final unit of space charge is in 1/(cm^3). This is in the same units as the simulation uses.
            frame_result[i][j] = space_charge
    return frame_result

# Wrapper around albert's function
def derivative_all(data: Tensor) -> Tensor:
    result = torch.zeros_like(data)
    x, y = load_spacing()
    for i in range(len(data)):
        result[i] = derivative_one_frame(data[i], x, y)
    return result

def load_normalize_space_charge() -> Tensor:
    return torch.tensor(load_space_charge()) * -Q

def load_derivative() -> Tensor:
    space_charge = load_normalize_space_charge()
    laplacian = derivative_all(torch.tensor(load_elec_potential())) * -Q
    laplacian[:, 0, :] = space_charge[:, 0, :]
    laplacian[:, -1, :] = space_charge[:, -1, :]
    laplacian[:, :, 0] = space_charge[:, :, 0]
    laplacian[:, :, -1] = space_charge[:, :, -1]
    return laplacian

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
def get_folder_directory(root: str, model: Model):
    # The base folder path
    folder_path = f"{root}/{model.name}"

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
    if os.path.isfile(f"{folder_path}/logs.txt"):
        with open(f"{folder_path}/logs.txt", 'r') as f:
            st = f.read()
        if st.strip() == model.logs.strip():
            return folder_path
    
    # Create something like Model (1), Model (2) and so on via recursion
    if '(' not in folder_path.split("/")[-1]:
        return get_folder_directory(f"{folder_path} (1)", model)
    
    num = int(folder_path.split("(")[1][:-1])
    path = folder_path.split("(")[0][:-1]
    return get_folder_directory(f"{path} ({num + 1})", model)
