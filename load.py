import numpy as np
import matplotlib.pyplot as plt
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
    heatmap = ax.imshow(data[0], cmap='hot')

    # Add a colorbar to the heatmap
    cbar = ax.figure.colorbar(heatmap, ax=ax)
    cbar.ax.set_ylabel('Intensity', rotation=-90, va="bottom")

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