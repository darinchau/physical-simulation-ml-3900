import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from matplotlib import animation
import torch
from torch.utils.data import Dataset, DataLoader

# Input a numpy array and a tuple and indexes everything in the indices of that tuple. May messes up order.
def index_include(data, include):
    return data[list(include)]

# Input a numpy array and a tuple and indexes everything except the indices of that tuple. May mess up data order
def index_exclude(data, exclude):
    return data[list(set(range(len(data))) - set(exclude))]

# Load and return data. We expect data to be some 3 dimensional np array (N, rows, cols).
def load_data_week_1():
    data = np.load("mesh_data_electrostatic_potential.npy")
    return data

def split_data(ins, train_idx):
    return index_include(ins, train_idx), index_exclude(ins, train_idx)

def wrap_data(ins, data, train_idx: tuple[int, ...]):
    num_data = len(data)
    class WrappedData(Dataset):
        def __init__(self, input, data, indices):
            self.idxs = np.array(indices)
            self.input = np.array(input)
            self.data = np.array(data)

        def __getitem__(self, index):
            idx = self.idxs[index]
            x = self.input[idx]
            y = self.data[idx]
            return x, y

        def __len__(self):
            return len(self.idxs)

    # Train data from 1, 11, 21, ..., 101
    train_data = WrappedData(ins, data, train_idx)

    # Test data from the others
    test_idx = tuple(set(range(num_data)) - set(train_idx))
    test_data = WrappedData(ins, data, test_idx)

    # Wrap in data loaders
    train_dl = DataLoader(train_data, batch_size=1, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=1, shuffle=False)

    return train_dl, test_dl

def make_anim(data, path = None):
    # Set up figure and axis for animation
    fig, ax = plt.subplots()
    heatmap = ax.imshow(data[0], cmap="hot")

    # Add a colorbar to the heatmap
    cbar = ax.figure.colorbar(heatmap, ax=ax)
    cbar.ax.set_ylabel("Intensity", rotation=-90, va="bottom")

    # Define update function for animation
    def update(frame):
        heatmap.set_data(data[frame])
        return heatmap,

    # Create animation object and display it
    anim = animation.FuncAnimation(fig, update, frames=data.shape[0], interval=50, blit=True)

    if path is not None:
        writergif = animation.PillowWriter(fps=30)
        anim.save(path, writer=writergif)
    else:
        plt.show()

def make_anim_week_2(predicted_data, original_data, path = None, prediction_name = None):
    # Calculate the error
    rmse = np.sqrt(np.mean((original_data - predicted_data)**2))
    worse = np.max(np.abs(original_data - predicted_data))

    spacing_x = np.load("mesh_data_x.npy")
    spacing_y = np.load("mesh_data_y.npy")

    x, y = np.meshgrid(spacing_x, spacing_y)

    # Set up figure and axis for animation
    fig, axes = plt.subplots(4, 2, gridspec_kw={'height_ratios': [4, 4, 4, 1], 'width_ratios': [7, 1]})

    # Figure title
    fig.suptitle(f"Results from {str(prediction_name)}")

    # First Color bar ranges for data and predicted heatmap
    vmin = min(np.min(original_data), np.min(predicted_data))
    vmax = min(np.max(original_data), np.max(predicted_data))

    # Ax 00 corresponds to the prediction and the data
    axes[0][0].set_aspect("equal", adjustable="box")
    axes[0][0].set_xlabel("X")
    axes[0][0].set_ylabel("Y", va="bottom")
    axes[0][0].set_title(f"Original data")
    axes[0][0].set_yticks([])
    data_heatmap = axes[0][0].pcolormesh(x, y, np.transpose(original_data[0]), cmap="hot", vmin=vmin, vmax=vmax)

    # Ax 10 for prediction
    axes[1][0].set_aspect("equal", adjustable="box")
    axes[1][0].set_xlabel("X")
    axes[1][0].set_ylabel("Y", va="bottom")
    axes[1][0].set_title(f"Predicted data")
    axes[1][0].set_yticks([])
    pred_heatmap = axes[1][0].pcolormesh(x, y, np.transpose(predicted_data[0]), cmap="hot", vmin=vmin, vmax=vmax)

    # Ax 1 for the data/prediction colorbar
    ax1 = [axes[0][1], axes[1][1]]
    cbar1 = Colorbar(ax1, cmap='hot', norm=Normalize(vmin=vmin, vmax=vmax))
    cbar1.ax.set_ylabel("Intensity", rotation=-90, va="bottom")

    # Ax 2- corresponds to the errors
    error = np.abs(predicted_data - original_data)
    axes[2][0].set_aspect("equal", adjustable="box")
    axes[2][0].set_xlabel("X")
    axes[2][0].set_ylabel("Y", va="bottom")
    axes[2][0].set_title(f"Error plot")
    axes[2][0].set_yticks([])
    err_heatmap = axes[2][0].pcolormesh(x, y, np.transpose(error[0]), cmap="hot", vmin=0, vmax=np.max(error))

    # Ax 21 for the error colorbar
    cbar2 = Colorbar(axes[2][0], err_heatmap)
    cbar2.ax.set_ylabel("Intensity", rotation=-90, va="bottom")

    # Ax 3 is used purely to display error bounds information
    axes[3][0].text(0, 0.5, f"RMSE Error = {rmse}, Worst error = {worse}", ha="left", va="center", fontsize=9)
    axes[3][0].set_axis_off()

    # Unused last axis
    axes[3][1].set_axis_off()

    # Define update function for animation
    def update(frame):
        data_heatmap.set_array(np.transpose(original_data[frame]))
        pred_heatmap.set_array(np.transpose(predicted_data[frame]))
        err_heatmap.set_array(np.transpose(error[frame]))
        return pred_heatmap, err_heatmap

    # Create animation object and display it
    anim = animation.FuncAnimation(fig, update, frames=predicted_data.shape[0], interval=50, blit=True)

    plt.tight_layout()

    if path is not None:
        writergif = animation.PillowWriter(fps=30)
        anim.save(path, writer=writergif)
    else:
        plt.show()

    plt.close("all")

    return rmse, worse