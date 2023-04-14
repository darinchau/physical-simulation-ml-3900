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
class AnimationMaker:
    def __init__(self, animation_length: int):
        self.maps = {}
        self.text_frames = []
        self.anim_len = animation_length

    # scale is for the scale of the color bar we might want to generate. Default is the min and max of the frame
    # name is the title of the plot
    def add_plot(self, data, name, scale = None):
        assert len(data) == self.anim_len
        self.maps[name] = (data, scale)
        return self

    def add_text(self, texts: str | list[str]):
        if isinstance(texts, str):
            texts = [texts for _ in range(self.anim_len)]
        assert len(texts) == self.anim_len
        self.text_frames.append(texts)
        return self

    def plot(self, suptitle, path):
        # Load the spacing using a mesh grid
        x, y = np.meshgrid(SPACING_X, SPACING_Y)

        # Number of axes to use
        num_axes = len(self.maps) + len(self.text_frames)
        height_ratios = [5] * len(self.maps) + [1] * len(self.text_frames)

        # Set up figure and axis for animation
        fig, axes = plt.subplots(num_axes, 1, gridspec_kw={'height_ratios': height_ratios})

        # Figure title
        fig.suptitle(suptitle)

        # Store the data maps somewhere
        heatmaps = {}

        i = 0
        for title, (data, scale) in self.maps.items():
            # Plot the first frame of everything
            axes[i].set_aspect("equal", adjustable="box")
            axes[i].set_xlabel("X")
            axes[i].set_ylabel("Y", va="bottom")
            axes[i].set_title(title)
            axes[i].set_yticks([])

            # If the scale is none, then use the min and max of the function
            # else if it is 'data', use the data's scale
            # else use the custom scale in the form of a tuple. If one of the entries is None, use the min/max of the data
            # We dont need custom scales now but we remain open here
            # This is where rust enums would be soooooooo helpful
            if scale is None:
                vmin, vmax = np.min(data), np.max(data)
            else:
                vmin, vmax = scale[0], scale[1]
                if vmin is None:
                    vmin = np.min(data)
                if vmax is None:
                    vmax = np.max(data)

            data_heatmap = axes[0].pcolormesh(x, y, np.transpose(data[0]), cmap = "hot", vmin = vmin, vmax = vmax)

            # colorbar
            cbar = fig.colorbar(data_heatmap, ax = axes[i])
            cbar.ax.set_ylabel("Intensity", rotation = -90, va = "bottom")
            tk0 = np.round(np.linspace(vmin, vmax, 4, endpoint=True), 2)
            cbar.set_ticks(tk0)

            # Increment the axis used
            i += 1

            # Store the heatmap for update later
            heatmaps[title] = data_heatmap

        # Now initialize the text
        text_artists = []
        for txt in self.text_frames:
            text_artist = axes[i].text(0, 0.5, txt[0], ha="left", va="center", fontsize=9)
            axes[i].set_axis_off()
            text_artists.append(text_artist)
            i += 1

        # Define update function for animation
        def update(frame):
            artists = []
            for k, heatmap in heatmaps.items():
                data, _ = self.maps[k]
                heatmap.set_array(np.transpose(data[frame]))
                artists.append(heatmap)

            for text, artist in zip(self.text_frames, text_artists):
                artist.set_text(text[frame])
                artists.append(artist)

            return artists

        # Create animation object and display it
        anim = animation.FuncAnimation(fig, update, frames = self.anim_len, interval=50, blit=True)

        plt.tight_layout()

        if path is not None:
            writergif = animation.PillowWriter(fps=20)
            anim.save(path, writer=writergif)
            optimize_gif(path)
        else:
            plt.show()

        plt.close("all")

# Import antics
__all__ = [
    "save_h5",
    "peek_h5",
    "split_data",
    "load_elec_potential",
    "index_exclude",
    "index_include",
    "optimize_gif",
    "AnimationMaker"
]
