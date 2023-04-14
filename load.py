import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, LogNorm
from matplotlib import animation
from math import ceil
import subprocess
from numpy.typing import NDArray
import h5py

SPACING_X = np.load("mesh_data_x.npy")
SPACING_Y = np.load("mesh_data_y.npy")
ELECTRIC_POTENTIAL = np.load("mesh_data_electrostatic_potential.npy")

def optimize_gif(path):
    # Uses the gifsicle library
    subprocess.call(['gifsicle', '-O3', '--optimize', path, '--colors', '256', '--output', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Input a numpy array and a tuple and indexes everything in the indices of that tuple. May messes up order.
def index_include(data, include):
    return data[list(include)]

# Input a numpy array and a tuple and indexes everything except the indices of that tuple. May mess up data order
def index_exclude(data, exclude):
    return data[list(set(range(len(data))) - set(exclude))]

# Load and return data. We expect data to be some 3 dimensional np array (N, rows, cols).
def load_elec_potential():
    return ELECTRIC_POTENTIAL

def split_data(ins, train_idx):
    return index_include(ins, train_idx), index_exclude(ins, train_idx)

### Save h5 files
def save_h5(d: dict[str, NDArray], path: str):
    # Open an HDF5 file in write mode
    with h5py.File(path, 'w') as f:

        # Enable compression with gzip and set compression level to 6
        f.attrs.create('compression', 'gzip')
        f.attrs.create('compression_level', 6)

        # Loop through dictionary keys and add them as groups to the file
        for key in d.keys():
            group = f.create_group(key)

            # Add the numpy array value to the group and enable compression
            dataset = group.create_dataset('data', data=d[key])
            dataset.attrs.create('compression', 'gzip')
            dataset.attrs.create('compression_level', 6)

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

# Wrapper class to help us plot data
def make_anim_week_3(predicted_data, original_data, path = None, prediction_name = None):
    spacing_x = np.load("mesh_data_x.npy")
    spacing_y = np.load("mesh_data_y.npy")

    x, y = np.meshgrid(spacing_x, spacing_y)

    # Set up figure and axis for animation
    fig, axes = plt.subplots(6, 1, gridspec_kw={'height_ratios': [5, 5, 5, 5, 1, 1]})

    # Figure title
    fig.suptitle(prediction_name)

    # Use the same color bar for data for predicted results as well
    orig_vim = np.min(original_data)
    orig_vmax = np.max(original_data)

    # =========================================================================
    # Ax 0 corresponds to the prediction and the data
    axn = 0
    # =========================================================================
    axes[axn].set_aspect("equal", adjustable="box")
    axes[axn].set_ylabel("Y", va="bottom")
    axes[axn].set_title(f"Original data")
    axes[axn].set_yticks([])
    data_heatmap = axes[axn].pcolormesh(x, y, np.transpose(original_data[0]), cmap="hot", vmin=orig_vim, vmax=orig_vmax)

    # colorbar
    cbar0 = fig.colorbar(data_heatmap, ax=axes[axn])
    cbar0.ax.set_ylabel("Intensity", rotation=-90, va="bottom")
    tk0 = np.round(np.linspace(orig_vim, orig_vmax, 4, endpoint=True), 2)
    cbar0.set_ticks(tk0)

    # =========================================================================
    # Ax 1 for prediction
    axn = 1
    # =========================================================================
    axes[axn].set_aspect("equal", adjustable="box")
    axes[axn].set_ylabel("Y", va="bottom")
    axes[axn].set_title(f"Predicted data")
    axes[axn].set_yticks([])
    pred_heatmap = axes[axn].pcolormesh(x, y, np.transpose(predicted_data[0]), cmap="hot", vmin=orig_vim, vmax=orig_vmax)

    # colorbar
    cbar1 = fig.colorbar(pred_heatmap, ax=axes[axn])
    cbar1.ax.set_ylabel("Intensity", rotation=-90, va="bottom")
    cbar1.set_ticks(tk0)

    # # =========================================================================
    # # Ax 2 corresponds to the errors
    axn = 2
    # # =========================================================================
    error = np.abs(predicted_data - original_data)
    axes[axn].set_aspect("equal", adjustable="box")
    axes[axn].set_ylabel("Y", va="bottom")
    axes[axn].set_title(f"Error plot")
    axes[axn].set_yticks([])
    err_heatmap = axes[axn].pcolormesh(x, y, np.transpose(error[0]), cmap="hot", vmin=0, vmax=np.max(error))

    # error colorbar. Set number of rounding decimals to one less than the order of magnitude
    cbar2 = fig.colorbar(err_heatmap, ax=axes[axn])
    cbar2.ax.set_ylabel("Intensity", rotation=-90, va="bottom")
    num_decimals = -ceil(np.log10(np.max(error))) + 1
    tk2 = np.round(np.linspace(0, np.max(error), 4, endpoint=True), num_decimals)
    cbar2.set_ticks(tk2)

    # =========================================================================
    # Ax 3 corresponds to the log plot of errors
    axn = 3
    # =========================================================================
    err_log_10 = np.log10(error)
    err_log_10[error < 1e-20] = np.min(err_log_10[error > 0]) - 1

    axes[axn].set_aspect("equal", adjustable="box")
    axes[axn].set_xlabel("X")
    axes[axn].set_ylabel("Y", va="bottom")
    axes[axn].set_title(f"Error log plot")
    axes[axn].set_yticks([])
    log_err_heatmap = axes[3].pcolormesh(x, y, np.transpose(err_log_10[0]), cmap="hot", vmin = np.min(err_log_10), vmax = np.max(err_log_10))

    # error colorbar. Set number of rounding decimals to one less than the order of magnitude
    cbar3 = fig.colorbar(log_err_heatmap, ax=axes[axn])
    cbar3.ax.set_ylabel("Intensity", rotation=-90, va="bottom")

    # =========================================================================
    # Calculate errors
    # =========================================================================
    rmse = np.sqrt(np.mean((predicted_data - original_data) ** 2))
    worst = np.max(np.abs(predicted_data - original_data))
    rmse_last_10_frames = np.sqrt(np.mean((predicted_data[-10:] - original_data[-10:]) ** 2))
    worst_last_10_frames = np.max(np.abs(predicted_data[-10:] - original_data[-10:]))

    # =========================================================================
    # Ax 4 is used purely to display error bounds information
    axn = 4
    # =========================================================================
    axes[axn].text(0, 0.5, f"RMSE: {round(rmse, 5)}, (last 10 frames): {round(rmse_last_10_frames, 5)}, worst = {round(worst, 5)} (last 10 frames) = {round(worst_last_10_frames, 5)}", ha="left", va="center", fontsize=9)
    axes[axn].set_axis_off()

    # =========================================================================
    # Ax 5 is frame number
    axn = 5
    # =========================================================================
    frame_nr = axes[axn].text(0, 0.5, f"Frame 0 - 0V", ha="left", va="center", fontsize=9)
    axes[axn].set_axis_off()

    # Define update function for animation
    def update(frame):
        data_heatmap.set_array(np.transpose(original_data[frame]))
        pred_heatmap.set_array(np.transpose(predicted_data[frame]))
        err_heatmap.set_array(np.transpose(error[frame]))
        log_err_heatmap.set_array(np.transpose(err_log_10[frame]))
        # The rounding is needed otherwise floating point antics makes everything look horrible
        frame_nr.set_text(f"Frame {frame} - {round(frame*0.0075, 4)}V")
        return data_heatmap, pred_heatmap, err_heatmap, log_err_heatmap, frame_nr

    # Create animation object and display it
    anim = animation.FuncAnimation(fig, update, frames=predicted_data.shape[0], interval=50, blit=True)

    plt.tight_layout()

    if path is not None:
        writergif = animation.PillowWriter(fps=20)
        anim.save(path, writer=writergif)
        # optimize_gif(path)
    else:
        plt.show()

    plt.close("all")
