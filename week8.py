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
def straight_line_score(data, algorithm, normalize = True):
    data = util.array(data)
    result = np.zeros((129, 17))
    for i in range(129):
        for j in range(17):
            if normalize:
                result[i, j] = util.straight_line_score_normalizing(data[:, i, j], algorithm)
            else:
                result[i, j] = util.straight_line_score(data[:, i, j], algorithm)
    return result

# The code used to generate the straight line plot
def plot_straightness(algorithm = 'linear', normalize = True):
    sc = load_space_charge() * -Q
    ep = load_elec_potential()

    sc_scores = straight_line_score(sc, algorithm, normalize).T
    ep_scores = straight_line_score(ep, algorithm, normalize).T

    plot_two_heatmaps(sc_scores, ep_scores)

    return sc_scores, ep_scores


def plot_two_heatmaps(a, b):
    x_spacing, y_spacing = load_spacing()
    x_grid, y_grid = np.meshgrid(x_spacing, y_spacing)

    fig, (ax, ax2) = plt.subplots(2, 1)

    vmin = np.min(a)
    vmax = np.max(a)
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylabel("Y", va="bottom")
    ax.set_yticks([])
    heatmap = ax.pcolormesh(x_grid, y_grid, a, cmap="hot", vmin = vmin, vmax = vmax)
    ax.set_title("Space charge straightness")
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.ax.set_ylabel("Straightness", rotation=-90, va="bottom")

    vmin = np.min(b)
    vmax = np.max(b)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_ylabel("Y", va="bottom")
    ax2.set_yticks([])
    heatmap = ax2.pcolormesh(x_grid, y_grid, b, cmap="hot", vmin = vmin, vmax = vmax)
    ax2.set_title("Electric potential straightness")
    cbar = fig.colorbar(heatmap, ax=ax2)
    cbar.ax.set_ylabel("Straightness", rotation=-90, va="bottom")

    fig.suptitle("Straight line score of plots")
    fig.show()
