from __future__ import annotations
import numpy as np
import torch
from load import load_spacing
from models_base import get_device
from numpy.typing import NDArray
from torch import Tensor, nn

__all__ = (
    "poisson_lhs",
    "poisson_rmse",
    "PoissonLoss"
)

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

def poisson_lhs(data: Tensor | NDArray, x: Tensor | NDArray, y: Tensor | NDArray) -> Tensor | NDArray:
    # Currently assumed that only silicon is everywhere
    # Later, will need to adjust for the silicon-oxide interface
    relative_permittivity_silicon = 11.7
    
    # Convert free space permittivity to F/cm
    e0_cm = (8.85418782e-12) / 100
    
    # Actual dielectric permittivity = permittivity of free space * permittivity of material
    # Initialize result array
    if isinstance(data, Tensor):
        eps = torch.fill(torch.zeros_like(data[:1]), e0_cm * relative_permittivity_silicon)
        result = torch.zeros_like(data)
    elif isinstance(data, np.ndarray):
        result = np.zeros_like(data)
        eps = np.full_like(data[:1], e0_cm * relative_permittivity_silicon)
    else:
        raise TypeError(f"data (type: {type(data).__name__}) is neither a Tensor or a numpy array")
    
    return laplacian(data, result, eps, x, y)

def normalized_poisson_mse_(data, space_charge, x, y):
    ep = data.reshape(-1, 129, 17)
    sc = space_charge.reshape(-1, 129, 17)
    ys = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15)
    lapla = poisson_lhs(ep, x, y)
    if isinstance(data, Tensor):
        return torch.mean((sc[:, 1:-1, ys] - lapla[:, 1:-1, ys]) ** 2)
    elif isinstance(data, np.ndarray):
        return np.mean((sc[:, 1:-1, ys] - lapla[:, 1:-1, ys]) ** 2)
    raise TypeError

def poisson_mse_(data, space_charge, x, y):
    """Returns a single number indicating the poisson rmse over the range profided"""
    q = 1.60217663e-19
    sc = space_charge * -q
    return normalized_poisson_mse_(data, sc, x, y)


def poisson_rmse(data: Tensor | NDArray, space_charge: Tensor | NDArray):
    x, y = load_spacing()
    if isinstance(data, Tensor):
        return torch.sqrt(poisson_mse_(data, space_charge, x, y))
    else:
        return np.sqrt(poisson_mse_(data, space_charge, x, y))

class PoissonLoss(nn.Module):
    """Gives the poisson equation - the value of ||∇²φ - (-q)S||
    where S is the space charge described in p265 of the PDF 
    https://www.researchgate.net/profile/Nabil-Ashraf/post/How-to-control-the-slope-of-output-characteristicsId-Vd-of-a-GAA-nanowire-FET-which-shows-flat-saturated-region/attachment/5de3c15bcfe4a777d4f64432/AS%3A831293646458882%401575207258619/download/Synopsis_Sentaurus_user_manual.pdf"""    
    def __init__(self):
        super().__init__()
        self.device = get_device()
        x, y = load_spacing()
        self.x = x.to(self.device)
        self.y = y.to(self.device)
    
    def forward(self, x, space_charge):
        return poisson_mse_(x, space_charge, self.x, self.y)

class NormalizedPoissonRMSE(PoissonLoss):
    """Normalized means we assume space charge has already been multiplied by -q
    Gives the poisson equation - the value of sqrt(||∇²φ - (-q)S||)
    where S is the space charge described in p265 of the PDF 
    https://www.researchgate.net/profile/Nabil-Ashraf/post/How-to-control-the-slope-of-output-characteristicsId-Vd-of-a-GAA-nanowire-FET-which-shows-flat-saturated-region/attachment/5de3c15bcfe4a777d4f64432/AS%3A831293646458882%401575207258619/download/Synopsis_Sentaurus_user_manual.pdf"""    
    def forward(self, x, space_charge):
        return torch.sqrt(poisson_mse_(x, space_charge, self.x, self.y))
