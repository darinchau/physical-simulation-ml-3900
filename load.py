import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
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

# Plots Predicted data, original data, error regions, errors and frame numbers
def plot_data_with_spacing(predicted_data, path = None, prediction_name = None):
    # Calculate error
    rmse = np.sqrt(np.average((predicted_data - ELECTRIC_POTENTIAL) ** 2))
    worst = np.max(np.abs(predicted_data - ELECTRIC_POTENTIAL))

    # Load spacings into mesh grid
    x, y = np.meshgrid(SPACING_X, SPACING_Y)

    # Set up figure and axis for animation
    fig, axes = plt.subplots(5, 1, gridspec_kw={'height_ratios': [5, 5, 5, 1, 1]})

    # Figure title
    fig.suptitle(f"Results from {str(prediction_name)}")

    # Use the same color bar for data for predicted results as well
    vmin = np.min(ELECTRIC_POTENTIAL)
    vmax = np.max(ELECTRIC_POTENTIAL)

    # Ax 0 corresponds to the prediction and the data
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y", va="bottom")
    axes[0].set_title(f"Original data")
    axes[0].set_yticks([])
    data_heatmap = axes[0].pcolormesh(x, y, np.transpose(ELECTRIC_POTENTIAL[0]), cmap="hot", vmin=vmin, vmax=vmax)

    # error colorbar
    cbar0 = fig.colorbar(data_heatmap, ax=axes[0])
    cbar0.ax.set_ylabel("Intensity", rotation=-90, va="bottom")
    tk0 = np.round(np.linspace(vmin, vmax, 4, endpoint=True), 2)
    cbar0.set_ticks(tk0)

    # Ax 1 for prediction
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y", va="bottom")
    axes[1].set_title(f"Predicted data")
    axes[1].set_yticks([])
    pred_heatmap = axes[1].pcolormesh(x, y, np.transpose(predicted_data[0]), cmap="hot", vmin=vmin, vmax=vmax)

    # error colorbar
    cbar1 = fig.colorbar(pred_heatmap, ax=axes[1])
    cbar1.ax.set_ylabel("Intensity", rotation=-90, va="bottom")
    cbar1.set_ticks(tk0)

    # Ax 2 corresponds to the errors
    error = np.abs(predicted_data - ELECTRIC_POTENTIAL)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y", va="bottom")
    axes[2].set_title(f"Error plot")
    axes[2].set_yticks([])
    err_heatmap = axes[2].pcolormesh(x, y, np.transpose(error[0]), cmap="hot", vmin=0, vmax=np.max(error))

    # error colorbar. Set number of rounding decimals to one less than the order of magnitude
    cbar2 = fig.colorbar(err_heatmap, ax=axes[2])
    cbar2.ax.set_ylabel("Intensity", rotation=-90, va="bottom")
    num_decimals = -ceil(np.log10(np.max(error))) + 1
    tk2 = np.round(np.linspace(0, np.max(error), 4, endpoint=True), num_decimals)
    cbar2.set_ticks(tk2)

    # Ax 3 is used purely to display error bounds information
    axes[3].text(0, 0.5, f"RMSE Error = {rmse}, Worst error = {worst}", ha="left", va="center", fontsize=9)
    axes[3].set_axis_off()

    # Ax 4 is used to display the frame number
    frame_nr = axes[4].text(0, 0.5, f"Frame 0 - 0V", ha="left", va="center", fontsize=9)
    axes[4].set_axis_off()

    # Define update function for animation
    def update(frame):
        data_heatmap.set_array(np.transpose(ELECTRIC_POTENTIAL[frame]))
        pred_heatmap.set_array(np.transpose(predicted_data[frame]))
        err_heatmap.set_array(np.transpose(error[frame]))
        # The rounding is needed otherwise floating point antics makes everything look horrible
        frame_nr.set_text(f"Frame {frame} - {round(frame*0.0075, 4)}V")
        return data_heatmap, pred_heatmap, err_heatmap, frame_nr

    # Create animation object and display it
    anim = animation.FuncAnimation(fig, update, frames=predicted_data.shape[0], interval=50, blit=True)

    plt.tight_layout()

    if path is not None:
        writergif = animation.PillowWriter(fps=20)
        anim.save(path, writer=writergif)
        optimize_gif(path)
    else:
        plt.show()

    plt.close("all")

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

# Import antics
__all__ = [
    "save_h5",
    "peek_h5",
    "plot_data_with_spacing",
    "split_data",
    "load_elec_potential",
    "index_exclude",
    "index_include",
    "optimize_gif"
]
