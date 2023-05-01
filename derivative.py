# This module is dedicated to calculate the derivative at every frame
import numpy as np
from numpy.typing import NDArray
from load import load_spacing
import torch

def split_data(data):
    center = data[1:-1, 1:-1]
    left = data[:-2, 1:-1]
    right = data[2:, 1:-1]
    up = data[1:-1, :-2]
    down = data[1:-1, 2:]
    return center, left, right, up, down

# def laplacian_center_albert(fx, x_coords, y_coords, epsilon):
#     E0 = 8.8541878128e-12
#     eps = 11.68 * E0

#     pc2, pl2, pr2, pu2, pd2 = split_data(fx)
#     pc0, pl0, pr0, pu0, pd0 = split_data(x_coords)
#     pc0, pl1, pr1, pu1, pd1 = split_data(y_coords)

#     ds = eps * ((pr2 - pc2) / (pr1 - pc1))

# Compute the Laplacian for one single frame
# Assumes the data is:
#        u
#        |
#   l -- c -- r
#        |
#        d
# Horizontally is x axis (axis 0)
# This does not take care of the boundary problems
def laplacian_center_one_frame(fx, x_coords, y_coords, epsilon):
    # Split the data
    fc, fl, fr, fu, fd = split_data(fx)
    xc, xl, xr, _, _ = split_data(y_coords)
    yc, _, _, yu, yd = split_data(x_coords)
    _, el, er, eu, ed = split_data(epsilon)

    # f_(x or y)_(coordinates) means calculate the first order finite difference at said point
    f_x_cl = (fc - fl)/(xc - xl)
    f_x_cr = (fc - fr)/(xc - xr)
    f_xx = (2 * el * f_x_cl - 2 * er * f_x_cr) / (xl - xr)

    f_y_cu = (fc - fu)/(yc - yu)
    f_y_cd = (fc - fd)/(yc - yd)
    f_yy = (2 * eu * f_y_cu - 2 * ed * f_y_cd) / (yu - yd)

    # Center part
    return f_xx + f_yy

# Calculate the laplacian for one frame
def laplacian_one_frame(data, x, y):
    # Permittivity of free space
    E0 = 8.8541878128e-12
    epsilon = torch.zeros_like(data) + 11.68 * E0
    return laplacian_center_one_frame(data, x, y, epsilon)

# With reference to Albert's formula and
# https://en.wikipedia.org/wiki/Finite_difference
def laplacian(data: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    """Compute the laplacian of data"""
    # Compute the laplacian
    result = torch.zeros_like(data)
    for i, d in enumerate(data):
        dPdx = torch.gradient(d, spacing = (x,), dim = 0, edge_order = 2)[0]
        dPdy = torch.gradient(d, spacing = (y,), dim = 1, edge_order = 2)[0]
        d2Pdx2 = torch.gradient(dPdx, spacing = (x,), dim = 0, edge_order = 2)[0]
        d2Pdy2 = torch.gradient(dPdy, spacing = (y,), dim = 1, edge_order = 2)[0]
        result[i] = d2Pdx2 + d2Pdy2
    
    return result
