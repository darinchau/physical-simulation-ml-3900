### Contains all the functions that are not directly implementations of models but dont fit in anywhere else
from __future__ import annotations
from typing import Any
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from numpy.typing import NDArray
from abc import abstractmethod as virtual
import torch

from load import load_spacing

def flip_mix(target):
    total = 0.077976
    x_spacing = load_spacing()[0]
    target_flipped = torch.zeros_like(target)
    
    for j in range(129):
        x = total - x_spacing[j]

        # Use normal flip near the edges
        if x < 0.02 or total - x < 0.02:
            col = np.abs(x - x_spacing).argmin()
            target_flipped[:, j, :] = target[:, col, :]
            continue
        
        # Use lerp near the center
        col1 = np.searchsorted(x_spacing, x, side='right') - 1
        col2 = col1 + 1
        weighting = (x_spacing[col2] - x)/(x_spacing[col2] - x_spacing[col1])
        target_flipped[:, j, :] = weighting * target[:, col1, :] + (1-weighting) * target[:, col2, :]
    return target_flipped
