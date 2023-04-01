import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
from torch.utils.data import Dataset, DataLoader

# Input a numpy array and a tuple and indexes everything except the indices of that tuple
def index_exclude(data, exclude):
    return data[list(set(range(len(data))) - set(exclude))]

# Load and return data. We expect data to be some 3 dimensional np array (N, rows, cols).
def load_data():
    data = np.load("mesh_data_electrostatic_potential.npy")
    print(f"Loaded data wiht shape {data.shape}")

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    return data, device

def wrap_data(input, data, train_idx: tuple[int, ...]):
    class WrappedData(Dataset):
        def __init__(self, input, data):
            self.input = np.array(input)
            self.data = np.array(data)

        def __getitem__(self, index):
            x = self.input[index]
            y = self.data[index]
            return x, y

        def __len__(self):
            return len(self.data)

    train_idx = tuple(train_idx)

    # Train data from 1, 11, 21, ..., 101
    train_data = WrappedData(input[train_idx,], data[train_idx, :, :])

    # Test data from the others
    test_data = WrappedData(index_exclude(input, train_idx), index_exclude(data, train_idx))

    # Wrap in data loaders
    train_dl = DataLoader(train_data, batch_size=1, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=1, shuffle=False)

    return train_dl, test_dl

def make_anim(data):
    # Set up figure and axis for animation
    fig, ax = plt.subplots()
    heatmap = ax.imshow(data[0], cmap='hot')

    # Define update function for animation
    def update(frame):
        heatmap.set_data(data[frame])
        return heatmap,

    # Create animation object and display it
    ani = animation.FuncAnimation(fig, update, frames=data.shape[0], interval=50, blit=True)
    plt.show()
