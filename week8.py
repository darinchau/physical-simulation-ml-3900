from load import *
from torch import Tensor, nn
import torch
from model_base import *
from modules import *
from anim import *
import util
import matplotlib.pyplot as plt

ROOT = "./Datas/Week 8"

Q = 1.60217663e-19

## This module contains everything miscellaneous generated during week 8
# I am now curious how "straight" is each region. 
def straight_line_score(data, algorithm):
    data = util.array(data)
    result = np.zeros((129, 17))
    for i in range(129):
        for j in range(17):
            result[i, j] = util.straight_line_score(data[:, i, j], algorithm)
    return result

# The code used to generate the straight line plot
def plot_straightness(algorithm = 'linear'):
    sc = load_space_charge() * -Q
    ep = load_elec_potential()
    x_spacing, y_spacing = load_spacing()
    x_grid, y_grid = np.meshgrid(x_spacing, y_spacing)

    fig, (ax, ax2) = plt.subplots(2, 1)

    sc_scores = straight_line_score(sc, algorithm).T
    ep_scores = straight_line_score(ep, algorithm).T
    vmax = 1

    vmin = np.min(sc_scores)
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylabel("Y", va="bottom")
    ax.set_yticks([])
    heatmap = ax.pcolormesh(x_grid, y_grid, sc_scores, cmap="hot", vmin = vmin, vmax = vmax)
    ax.set_title("Space charge straightness")
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.ax.set_ylabel("Straightness", rotation=-90, va="bottom")

    vmin = np.min(ep_scores)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_ylabel("Y", va="bottom")
    ax2.set_yticks([])
    heatmap = ax2.pcolormesh(x_grid, y_grid, ep_scores, cmap="hot", vmin = vmin, vmax = vmax)
    ax2.set_title("Electric potential straightness")
    cbar = fig.colorbar(heatmap, ax=ax2)
    cbar.ax.set_ylabel("Straightness", rotation=-90, va="bottom")

    fig.suptitle("Straight line score of plots")
    fig.show()
