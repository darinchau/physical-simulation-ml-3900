import os
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

def load_log_e_density():
    log_e_density = load_e_density()
    log_e_density[log_e_density < 1] = 1
    log_e_density = np.log10(log_e_density)
    return log_e_density

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

# A class wrapper to help us plot data. We assume all error checking has been done
class AnimationMaker:
    def __init__(self) -> None:
        # Initialize the maker. We will use height ratios to sort of keep track of the types of the data
        self.datas = []
        self.height_ratios = []
        self.nframes = None

    def add_data(self, data, title, vmin = None, vmax = None):
        if self.nframes is None:
            self.nframes = len(data)
        else:
            assert len(data) == self.nframes

        if vmin is None:
            vmin = np.min(data)
        if vmax is None:
            vmax = np.max(data)
        self.datas.append((np.array(data), title, vmin, vmax))
        self.height_ratios.append(5)
        return self


    def add_text(self, text):
        if self.nframes is None:
            self.nframes = len(text)
        else:
            assert len(text) == self.nframes

        self.datas.append(text)
        self.height_ratios.append(1)
        return self

    def save(self, path, suptitle = ""):
        if self.nframes is None:
            raise ValueError("No data to plot")

        # Create a mesh_grid
        x, y = np.meshgrid(SPACING_X, SPACING_Y)

        # Create a list of artists to update
        artists = []

        # Number of datas to plot
        num_data = len(self.datas)

        # Set up figure and axis for animation
        fig, axes = plt.subplots(num_data, 1, gridspec_kw={'height_ratios': self.height_ratios})

        # Figure title
        fig.suptitle(suptitle)

        # One axis one data
        for i in range(num_data):
            # Plot an animation
            if self.height_ratios[i] == 5:
                data, title, vmin, vmax = self.datas[i]

                axes[i].set_aspect("equal", adjustable="box")
                axes[i].set_ylabel("Y", va="bottom")
                axes[i].set_title(title)
                axes[i].set_yticks([])
                heatmap = axes[i].pcolormesh(x, y, np.transpose(data[0]), cmap="hot", vmin=vmin, vmax=vmax)

                # colorbar
                cbar = fig.colorbar(heatmap, ax=axes[i])
                cbar.ax.set_ylabel("Intensity", rotation=-90, va="bottom")
                tk = np.round(np.linspace(vmin, vmax, 4, endpoint=True), 2)
                cbar.set_ticks(tk)

                # Save the heatmap somewhere
                artists.append(heatmap)

            # Text data
            else:
                data = self.datas[i]
                text = axes[i].text(0, 0.5, data[0], ha="left", va="center", fontsize=9)
                axes[i].set_axis_off()

                # Save the text somewhere
                artists.append(text)

        # Update the color maps
        def update(frame):
            for i in range(num_data):
                if self.height_ratios[i] == 5:
                    heatmap = artists[i]
                    fr = self.datas[i][0][frame]
                    heatmap.set_array(np.transpose(fr))
                else:
                    text = artists[i]
                    text.set_text(self.datas[i][frame])
            return artists

        # Create animation object and display it
        anim = animation.FuncAnimation(fig, update, frames=self.nframes, interval=50, blit=True)

        plt.tight_layout()

        if path is not None:
            writergif = animation.PillowWriter(fps=15)
            anim.save(path, writer=writergif)
            # optimize_gif(path)
        else:
            plt.show()

        plt.close("all")

# Wrapper function to help us plot data
def make_anim(predicted_data, original_data, path = None, title = None):
    anim = AnimationMaker()

    data_vmin, data_vmax = np.min(original_data), np.max(original_data)
    anim.add_data(original_data, "Original data")

    anim.add_data(predicted_data, "Predicted data", data_vmin, data_vmax)

    error = np.abs(predicted_data - original_data)
    anim.add_data(error, "Error", 0)

    err_log_10 = np.log10(error)
    err_log_10[error < 1e-20] = np.min(err_log_10[error > 0]) - 1
    anim.add_data(err_log_10, "Error log plot")

    # Errors
    rmse = np.sqrt(np.mean((predicted_data - original_data) ** 2))
    worst = np.max(np.abs(predicted_data - original_data))
    rmse_last_10_frames = np.sqrt(np.mean((predicted_data[-10:] - original_data[-10:]) ** 2))
    worst_last_10_frames = np.max(np.abs(predicted_data[-10:] - original_data[-10:]))

    error_text = f"RMSE: {round(rmse, 5)}, (last 10 frames): {round(rmse_last_10_frames, 5)}, worst = {round(worst, 5)} (last 10 frames) = {round(worst_last_10_frames, 5)}"
    anim.add_text([error_text for _ in range(101)])

    # Frame number
    frames = [f"Frame {frame} - {round(frame*0.0075, 4)}V" for frame in range(101)]
    anim.add_text(frames)

    anim.save(path, title)

# Helper function to extract the middle region and outer region
def split_mid_outer(data):
    middle = data[:, 40:-40, :]
    outer = np.concatenate([data[:, :40, :], data[:, -40:, :]], axis=1)
    return middle, outer

# Helper function that creates the line graphs across frames for different n
def make_static_plot(frame_errors, val_f, plot_name, path):
        # Plot error each frame
        fig, ax = plt.subplots()

        # Loop through all the frame errors
        for key, list_values in frame_errors.items():
            if key[0] == "_":
                continue
            
            ax.plot([val_f(entry) for entry in list_values], label=key)

        # add legend to the plot
        ax.legend()

        # Title
        fig.suptitle(f"{plot_name} for each frame")

        # Show the thing
        fig.savefig(f"{path}/{plot_name}.png")

        try:
            # Set y-axis to log scale
            ax.set_yscale('log')

            # Save the figure again in log scale
            fig.savefig(f"{path}/{plot_name} log.png")
        except ValueError:
            pass

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
            # Prediction, the "take everythign" slice index is to change it to numpy array
            pred = f[key]['data'][:]

            # First plot is the animation
            # Animation :D
            make_anim(pred, original_data, f"{path}/{key}.gif", f"Results from {model_name} {key}")

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

        make_static_plot(frame_errors, lambda x: x[0], f"{model_name} RMSE Error", path)
        make_static_plot(frame_errors, lambda x: x[1], f"{model_name} Worst Error", path)
        make_static_plot(frame_errors, lambda x: x[2], f"{model_name} Middle RMSE", path)
        make_static_plot(frame_errors, lambda x: x[3], f"{model_name} Outer RMSE", path)
