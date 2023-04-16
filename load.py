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
ELECTRON_DENSITY = np.load("mesh_data_edensity.npy")

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
    return np.array(ELECTRIC_POTENTIAL)

def load_e_density():
    return np.nan_to_num(ELECTRON_DENSITY, nan=0)

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
def make_anim(predicted_data, original_data, path = None, prediction_name = None):
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

# Helper function to extract the middle region and outer region
def split_mid_outer(data):
    middle = data[:, 40:-40, :]
    outer = np.concatenate([data[:, :40, :], data[:, -40:, :]], axis=1)
    return middle, outer

# Helper function that creates the line graphs across frames for different n
def make_static_plot(frame_errors, val_f, plot_name, path):
        # Plot error each frame
        fig, ax = plt.subplots()

        for key, list_values in frame_errors.items():
            # The indexing is on keys which is of the format "frame 123"
            # So all it does is to crop away the prepend
            if int(key[6:]) in (1, 5, 10, 20, 40, 60, 90):
                ax.plot([val_f(entry) for entry in list_values], label=f"First {key[6:]}")
            else:
                continue

        # add legend to the plot
        ax.legend()

        # Title
        fig.suptitle(f"{plot_name} using the first n data across frames")

        # Show the thing
        fig.savefig(f"{path}/{plot_name} across frame.png")

        # Set y-axis to log scale
        ax.set_yscale('log')

        # Save the figure again in log scale
        fig.savefig(f"{path}/{plot_name} across frame log.png")

# Takes a path, reads the predictions inside and generate all sorts of animations/plots
# If you want to add extra plots, this is the function you have to worry about
def make_plots(path, model_name = None):
    # Retreive the model name from the path if the user did not provide explicitly
    if model_name is None:
        model_name = path.split("/")[-1]

    with h5py.File(f"{path}/predictions.h5", 'r') as f:
        # Keep note of the frame errors
        frame_errors = {k: [] for k in f.keys()}

        # Loop through keys of the file and print them
        original_data = load_elec_potential()
        data_mid, data_outer = split_mid_outer(original_data)

        for key in f.keys():
            # Prediction, the index is to change it to numpy array
            pred = f[key]['data'][:]

            # First plot is the animation
            # Animation :D
            make_anim(pred, original_data, f"{path}/first {key[6:]}.gif", f"Results from {model_name} first {key[6:]}")

            pred_mid, pred_outer = split_mid_outer(pred)

            # Second plot is error each frame for different ns
            # Uses a for loop to save memory. I know einsum is a thing but I dont know how to use it
            for i in range(101):
                # General RMSE
                rmse = np.sqrt(np.mean((pred[i] - original_data[i]) ** 2))
                worst = np.max(np.abs(pred[i] - original_data[i]))

                # Region-specific RMSE
                middle_rmse = np.sqrt(np.mean((pred_mid[i] - data_mid[i]) ** 2))
                outer_rmse = np.sqrt(np.mean((pred_outer[i] - data_outer[i]) ** 2))

                # Append all errors
                frame_errors[key].append((rmse, worst, middle_rmse, outer_rmse))

            make_static_plot(frame_errors, lambda x: x[0], "RMSE Error", path)
            make_static_plot(frame_errors, lambda x: x[1], "Worst Error", path)
            make_static_plot(frame_errors, lambda x: x[2], "Middle RMSE", path)
            make_static_plot(frame_errors, lambda x: x[3], "Outer RMSE", path)
