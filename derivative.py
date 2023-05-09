import numpy as np
import torch
from load import load_spacing
from numpy.typing import NDArray

# Wrapper around albert's function, without the division by -Q
def laplacian(data, result, eps, x, y):
    """Calculate the LHS of poisson equation"""
    xr, yr, epr = x[1:-1].view(-1, 1), y[2:], data[:, 1:-1, 2:]
    xd, yd, epd = x[2:].view(-1, 1), y[1:-1], data[:, 2:, 1:-1]
    xc, yc, epc = x[1:-1].view(-1, 1), y[1:-1], data[:, 1:-1, 1:-1]
    xl, yl, epl = x[1:-1].view(-1, 1), y[:-2], data[:, 1:-1, :-2]
    xu, yu, epu = x[:-2].view(-1, 1), y[1:-1], data[:, :-2, 1:-1]

    eps_r = eps[:, 1:-1, 2:]
    eps_d = eps[:, 2:, 1:-1]
    eps_c = eps[:, 1:-1, 1:-1]
    eps_l = eps[:, 1:-1, :-2]
    eps_u = eps[:, :-2, 1:-1]

    dx_right = eps_r * (epr - epc) / (yr - yc)
    dx_left = eps_l * (epc - epl) / (yc - yl)
    dxx = 2 * (dx_right - dx_left) / (yr - yl)

    dy_up = eps_u * (epu - epc) / (xu - xc)
    dy_down = eps_d * (epc - epd) / (xc - xd)
    dyy = 2 * (dy_up - dy_down) / (xu - xd)
    
    # the final unit of "ds" is in C/(cm * um^2). Multiply by 1e8 to convert the um^2 to cm^2 to get to C/(cm^3).
    div_cm = (dxx + dyy) * 1e8
    result[:, 1:-1, 1:-1] = div_cm
    return result

def laplacian_all(data: torch.Tensor) -> torch.Tensor:
    x, y = load_spacing()
    result = torch.zeros_like(data)
    # Currently assumed that only silicon is everywhere
    # Later, will need to adjust for the silicon-oxide interface
    relative_permittivity_silicon = 11.7
    
    # Convert free space permittivity to F/cm
    e0_cm = (8.85418782e-12) / 100
    
    # Actual dielectric permittivity = permittivity of free space * permittivity of material
    eps = torch.fill(torch.zeros_like(data[:1]), e0_cm * relative_permittivity_silicon)
    return laplacian(data, result, eps, x, y)

def laplacian_all_numpy(data: NDArray) -> NDArray:
    x, y = load_spacing()
    result = np.zeros_like(data)
    relative_permittivity_silicon = 11.7
    e0_cm = (8.85418782e-12) / 100
    eps = np.full_like(data[:1], e0_cm * relative_permittivity_silicon)
    return laplacian(data, result, eps, x, y)
