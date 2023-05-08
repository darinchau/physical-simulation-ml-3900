import numpy as np
import torch
from anim import AnimationMaker, log_diff, DataVisualizer
from load import load_spacing, load_elec_potential, load_space_charge

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

def laplacian_all(data):
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

def test():
    q = 1.60217663e-19

    anim = AnimationMaker()
    space_charge = torch.tensor(load_space_charge()) * -q
    anim.add_data(space_charge, "Space charge")

    lapla = laplacian_all(load_elec_potential())
    lapla[:, 0, :] = space_charge[:, 0, :]
    lapla[:, -1, :] = space_charge[:, -1, :]
    lapla[:, :, 0] = space_charge[:, :, 0]
    lapla[:, :, -1] = space_charge[:, :, -1]
    anim.add_data(lapla, "Laplacian 2", vmin = torch.min(space_charge), vmax = torch.max(space_charge))

    # dv = DataVisualizer(log_diff(space_charge, lapla))
    # dv.add_data(space_charge, "Space charge")
    # dv.add_data(lapla, "Laplacian")
    # dv.show()

    anim.add_data(log_diff(space_charge, lapla), "Log difference", vmin = -8)
    anim.add_data(log_diff(space_charge, lapla), "Log difference", vmin = -5)
    anim.add_text([f"Frame {i}" for i in range(101)])
    anim.save("derivatives.gif")

    print(f"RMSE: {torch.sqrt(torch.mean((space_charge - lapla) ** 2))}")
    print(f"RMSE under cut line: {torch.sqrt(torch.mean((space_charge[:, 1:-1, :10] - lapla[:, 1:-1, :10]) ** 2))}")
    print(f"RMSE on cut line: {torch.sqrt(torch.mean((space_charge[:, 1:-1, 10] - lapla[:, 1:-1, 10]) ** 2))}")
    print(f"RMSE over cut line: {torch.sqrt(torch.mean((space_charge[:, 1:-1, 11:] - lapla[:, 1:-1, 11:]) ** 2))}")

# Given one single frame of x and space charge, compare the derivatives :)
def compare_derivative(x, space_charge):
    q = 1.60217663e-19
    ep = x.reshape(-1, 129, 17)
    sc = space_charge.reshape(-1, 129, 17) * -q
    # Skip the cut line and the boundary conditions
    ys = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15)
    lapla = laplacian_all(ep)
    rmse = torch.sqrt(torch.mean((sc[:, 1:-1, ys] - lapla[:, 1:-1, ys]) ** 2))
    return rmse

def test_2():
    potential = load_elec_potential()
    space_charge = load_space_charge()

    frames = torch.zeros_like(space_charge)

    for i in range(101):
        rmse = compare_derivative(potential[i], space_charge[i])
        print(f"RMSE under frame: {rmse}")

if __name__ == "__main__":
    test()
    # test_2()