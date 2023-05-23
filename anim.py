### This module contains all methods for making the plots and animations in our project ###
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import animation
import numpy as np
from load import load_spacing, load_elec_potential, load_space_charge
import h5py
from torch import Tensor
import torch
from derivative import poisson_rmse
from numpy.typing import NDArray
from sklearn.linear_model import TheilSenRegressor
import util
from typing import Callable
from tqdm import tqdm
from multiprocessing import Pool
import warnings

# A class wrapper to help us plot data. We assume all error checking has been done
class AnimationMaker:
    def __init__(self) -> None:
        # Initialize the maker. We will use height ratios to sort of keep track of the types of the data
        self.datas = []
        self.height_ratios = []
        self.nframes = None

    def add_data(self, data, title: str, vmin: int | float | None = None, vmax: int | float | None = None, norm: str | None = None):
        if self.nframes is None:
            self.nframes = len(data)
        else:
            assert len(data) == self.nframes

        data = util.array(data)
        
        if vmin is None:
            vmin = np.min(data)
        if vmax is None:
            vmax = np.max(data)
        self.datas.append((data, title, vmin, vmax))
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

    def save(self, path, suptitle = "", cmap = 'hot'):
        # If nframes if not set this means there is no data to plot
        if self.nframes is None:
            raise ValueError("No data to plot")
        
        # Create a mesh_grid
        x, y = load_spacing()
        x, y = np.meshgrid(x, y)

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
                
                heatmap = axes[i].pcolormesh(x, y, np.transpose(data[0]), cmap=cmap, vmin=vmin, vmax=vmax)

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
        else:
            plt.show()

        plt.close("all")

def make_dem_animation(ypred, y) -> AnimationMaker:
    # Make animation for showcasing
    anim = AnimationMaker()

    ymin, ymax = np.min(y), np.max(y)
    anim.add_data(y, "Original data")

    anim.add_data(ypred, "Predicted data (rescaled)", ymin, ymax)

    error = np.abs(ypred - y)
    anim.add_data(error, "Error", 0)

    err_log_10 = log_diff(ypred, y)
    anim.add_data(err_log_10, "Error log plot")

    rmse = np.sqrt(np.mean((ypred - y) ** 2))
    worst = np.max(np.abs(ypred - y))
    rmse_last_10_frames = np.sqrt(np.mean((ypred[-10:] - y[-10:]) ** 2))
    worst_last_10_frames = np.max(np.abs(ypred[-10:] - y[-10:]))

    error_text = f"RMSE: {round(rmse, 5)}, (last 10 frames): {round(rmse_last_10_frames, 5)}, worst = {round(worst, 5)} (last 10 frames) = {round(worst_last_10_frames, 5)}"
    anim.add_text([error_text for _ in range(101)])

    frames = [f"Frame {frame} - {round(frame*0.0075, 4)}V" for frame in range(101)]
    anim.add_text(frames)
    return anim

def make_debug_animation(ypred, y) -> AnimationMaker:
    # Make animation for debug
    anim = AnimationMaker()
    anim.add_data(y, "Original data")
    anim.add_data(ypred, "Predicted data")

    frame_zero = np.zeros((101, 1, 1)) + ypred[0]
    diff_log_10 = log_diff(ypred, frame_zero)
    anim.add_data(diff_log_10, "Evolution (log)")
    
    anim.add_text([f"RMSE: {np.sqrt(np.mean((ypred[i] - y[i]) ** 2)):.7f}" for i in range(101)])
    anim.add_text([f"Worst: {np.max(np.abs(ypred[i] - y[i])):.7f}" for i in range(101)])
    
    space_charge: NDArray = load_space_charge().cpu().numpy()
    anim.add_text([f"Poisson RMSE: {poisson_rmse(ypred[i:i+1], space_charge[i:i+1]):.7f}" for i in range(101)])

    anim.add_text([f"Frame {frame} - {round(frame*0.0075, 4)}V" for frame in range(101)])
    return anim

# Wrapper function to help us plot data
def make_anim(predicted_data, original_data, path = None, title = None):
    if path is None:
        import time
        path = f"./{time.time_ns()}.gif"
    if title is None:
        title = "Animation"
    ypred = util.array(predicted_data)
    ytest = util.array(original_data)
    a1 = make_dem_animation(ypred, ytest)
    a2 = make_debug_animation(ypred, ytest)
    a1.save(path, title)
    a2.save(f"{path[:-4]} debug.gif", title + " debug")

# Space charge has a weird structure
# --------------------------------------------------------------------
# -                                                                  -
# -       Region 4                Region 5            Region 6       -
# 11                      44 45              83 84                   -
# 10 -----------------------                    ----------------------
# -                        -                    -                    -
# -       Region 1         -      Region 2      -      Region 3      -
# -                        -                    -                    -
# --------------------------------------------------------------------
# mid means region 2, 5
def split_mid_outer(data):
    middle = data[:, 45:84, :]
    outer = np.concatenate([data[:, :45, :], data[:, 84:, :]], axis=1)
    return middle, outer
        
# Wrapper for static plots maker
class StaticPlotMaker:
    def __init__(self):
        self.infos: dict[str, list] = {}
        
    def set(self, key: str, info):
        self.infos[key] = list(info)
    
    def append(self, key: str, value):
        if key not in self.infos:
            self.infos[key] = []
        self.infos[key].append(value)
    
    def plot(self, plot_name, path, title = None):
        # Make the title
        if title is None:
            title = f"{plot_name} for each frame"

        fig, ax = plt.subplots()

        for key, list_values in self.infos.items():
            ax.plot(list_values, label=key)

        ax.legend()
        fig.suptitle(title)
        fig.savefig(f"{path}/{plot_name}.png")

        try:
            ax.set_yscale('log')
            fig.savefig(f"{path}/{plot_name} log.png")
        except ValueError:
            pass

# Takes a path, reads the predictions inside and generate all sorts of animations/plots
# If you want to add extra plots, this is the function you have to worry about
# include_in_error is a variable which contains a list of strings to include in the error plots
# If none, then include everything
def make_plots(path, model_name = None, include_keys: list[str] | None = None):
    """Main function for plots creation"""
    # Retreive the model name from the path if the user did not provide explicitly
    if model_name is None:
        model_name = path.split("/")[-1]
    
    # Keep note of the frame errors
    frame_rmse = StaticPlotMaker()
    frame_rmse_outer = StaticPlotMaker()
    frame_rmse_mid = StaticPlotMaker()
    frame_worst = StaticPlotMaker()
    frame_mid_potential = StaticPlotMaker()
    
    # Loop through keys of the file and print them
    original_data = util.array(load_elec_potential())
    data_mid, data_outer = split_mid_outer(original_data)
    
    # Plot the middle potential for the original data
    frame_mid_potential.set("Original", np.average(data_mid.reshape((101, -1)), axis = 1))

    with h5py.File(f"{path}/predictions.h5", 'r') as f:
        # Make a customary error message if we happen to make a typo in the list
        if include_keys is not None:
            for key in include_keys:
                if key not in f.keys():
                    print(f"The key {key} is not found in the file for {model_name}.")
        
        # Loop through all the keys to make the animation and calculate the error
        for key in f.keys():
            # Prediction, the "take everything" slice index is to change it to numpy array
            pred = f[key]['data'][:]
            
            # Animation :D
            make_anim(pred, original_data, f"{path}/{key}.gif", f"Results from {model_name} {key}")
            
            # If we indicate to not appear in plot, then skip everything else here
            # If include_in_error is none that means include everything
            if include_keys is not None and key not in include_keys:
                continue

            # Split the middle region and outer region
            pred_mid, pred_outer = split_mid_outer(pred)

            # Second plot is error each frame for different ns
            # Uses a for loop to save memory
            for i in range(101):
                # General RMSE
                rmse = np.sqrt(np.mean((pred[i] - original_data[i]) ** 2))
                frame_rmse.append(key, rmse)
                
                # Worst error
                worst = np.max(np.abs(pred[i] - original_data[i]))
                frame_worst.append(key, worst)

                # Region-specific RMSE
                middle_rmse = np.sqrt(np.mean((pred_mid[i] - data_mid[i]) ** 2))
                frame_rmse_mid.append(key, middle_rmse)
                
                # Region-specific RMSE. THe outer one is usually not interesting
                outer_rmse = np.sqrt(np.mean((pred_outer[i] - data_outer[i]) ** 2))
                frame_rmse_outer.append(key, outer_rmse)
                
                # The middle growth thing
                mid_potential = np.average(pred_mid[i])
                frame_mid_potential.append(key, mid_potential)
        
        frame_rmse.plot(f"{model_name} RMSE Error", path)
        frame_worst.plot(f"{model_name} Worst Error", path)
        frame_rmse_mid.plot(f"{model_name} Middle RMSE", path)
        frame_rmse_outer.plot(f"{model_name} Outer RMSE", path)
        frame_mid_potential.plot(f"{model_name} mid potential", path, title = f"Electric potential of gate region from {model_name}")

class DataVisualizer:
    def __init__(self, cover = None) -> None:
        self.datas = {}
        if cover is None:
            self.datas["Original"] = (util.array(load_elec_potential()), 3)
        else:
            self.datas["Original"] = (util.array(cover), 3)
    
    def add_data(self, data, name):
        self.datas[name] = (util.array(data), 1)
    
    def show(self):        
        # Load the spacing
        x_spacing, y_spacing = load_spacing()
        x_grid, y_grid = np.meshgrid(x_spacing, y_spacing)

        # Handle mouse click
        def onclick(event, ax):
            # Do a preliminary check for None values
            if event.xdata is None:
                return
            
            if event.ydata is None:
                return
            
            if event.inaxes != ax:
                return
            
            # Get the row and column indices of the clicked cell
            x, y = event.xdata, event.ydata

            # Find the index of the nearest spacing value
            row = np.abs(y - y_spacing).argmin()
            col = np.abs(x - x_spacing).argmin()

            # Plot the values
            fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [6, 6, 1]})

            # Get the values of the cell across all arrays
            for k, (v, t) in self.datas.items():
                values = v[:, col, row]
                ax[0].plot(values, label = k, linewidth = t)
                
                if k == "Original":
                    continue
                
                cover = self.datas["Original"][0]
                orig_v = cover[:, col, row]
                ax[1].plot(np.abs(values - orig_v), label = k)
            
            ax[1].set_xlabel("Array index")
            
            ax[0].set_ylabel("Potential")
            ax[1].set_ylabel("Log Error")
            
            ax[0].legend()
            ax[1].legend()
            
            ax[1].set_yscale('log')
            
            # Make the text
            y = self.datas["Original"][0][:, col, row]
            r_score = util.straight_line_score_normalizing(y)

            coord_info = f"Cell ({row}, {col}), Position: x = {event.xdata:.5f}, y = {event.ydata:.5f}"
            line_info = f"Straightness of original: {r_score:.5f}"

            ax[2].text(0, 0.5, "\n".join([coord_info, line_info]), ha="left", va="center", fontsize=9)
            ax[2].set_axis_off()
            
            fig.tight_layout()
            fig.show()

        # Create the heatmap
        fig, ax = plt.subplots()

        cover = self.datas["Original"][0]

        ax.set_aspect("equal", adjustable="box")
        ax.set_ylabel("Y", va="bottom")
        ax.set_yticks([])
        heatmap = ax.pcolormesh(x_grid, y_grid, np.transpose(cover[0]), cmap="hot", vmin = np.min(cover), vmax = np.max(cover))

        cbar = fig.colorbar(heatmap, ax=ax)
        cbar.ax.set_ylabel("Intensity", rotation=-90, va="bottom")
        tk = np.round(np.linspace(np.min(cover), np.max(cover), 4, endpoint=True), 2)
        cbar.set_ticks(tk)

        # Add a title
        fig.suptitle("Visualize data")

        # Add the click event handler
        cid = fig.canvas.mpl_connect("button_press_event", lambda event: onclick(event, ax))

        # Show the plot
        plt.show()

def log_diff(x, y):
    if isinstance(x, Tensor):
        x = x.detach().numpy()
    
    if isinstance(y, Tensor):
        y = y.detach().numpy()

    warnings.filterwarnings('ignore', category=RuntimeWarning)
    result = np.log10(np.abs(x - y))
    warnings.filterwarnings('default', category=RuntimeWarning)
    x = np.nanmin(result[np.isfinite(result)])
    result = np.nan_to_num(result, nan=x, posinf=x, neginf=x)
    return result


def heatmat_fn_plot(f: Callable[[float, float], float], xrange: tuple[int, int] = (-5, 5), yrange: tuple[int, int] = (-5, 5), resolution = 100):
    """Plot the function as a heatmap.
    - `xrange/yrange`: specify the x and y range of the plot
    - `resolution`: specify the number of subdivisions in a plot
    - `multiprocess`: if > 0, then use n processes. Otherwise use one single process"""
    # Generate x and y values
    x = np.linspace(xrange[0], xrange[1], resolution)
    y = np.linspace(yrange[0], yrange[1], resolution)

    # Create empty array for Z values
    Z = np.empty((len(x), len(y)))

    # Calculate the corresponding Z values using the function f
    args = [(x_val, y_val) for x_val in x for y_val in y]
    results = [(a[0], a[1], f(a[0], a[1])) for a in tqdm(args)]
    

    min_x, min_y, min_f = 0, 0, 9999
    for x_val, y_val, r in results:
        Z[np.where(x == x_val), np.where(y == y_val)] = r
        if r < min_f:
            min_x, min_y, min_f = x_val, y_val, r
    
    print(f"Minimum x: {min_x}, Minimum y: {min_y}, minimum f: {min_f}")

    # Create the heatmap plot
    fig, ax = plt.subplots()
    im = ax.imshow(Z, cmap='hot', extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("F(x, y) value", rotation=-90, va="bottom")

    # Set plot title and labels
    fig.suptitle("Heatmap Plot of f(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return fig, ax, Z
