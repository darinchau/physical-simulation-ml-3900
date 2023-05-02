import numpy as np
import torch
from anim import AnimationMaker, log_diff
from load import load_spacing, load_elec_potential, load_space_charge

def derivative_one_frame(data, x, y):
    frame_result = torch.zeros_like(data)

    # Ignores calculating at edges of the array
    for i in range(1, len(data) - 2):
        for j in range(1, len(data[0]) - 1):
            # unit of "eps_*": F/cm = C/(V*cm) 
            # unit of x and y are in um (converted to cm later)
            # unit of electrostatic potential is in V
            xr, yr, epr = x[i], y[j + 1], data[i][j + 1]
            xd, yd, epd = x[i + 1], y[j], data[i + 1][j]
            xc, yc, epc = x[i], y[j], data[i][j]
            xl, yl, epl = x[i], y[j - 1], data[i][j - 1]
            xu, yu, epu = x[i - 1], y[j], data[i - 1][j]
            
            # Currently assumed that only silicon is everywhere
            # Later, will need to adjust for the silicon-oxide interface
            relative_permittivity_silicon = 11.7
            
            # Convert free space permittivity to F/cm
            e0_cm = (8.85418782e-12) / 100
            
            # Actual dielectric permittivity = permittivity of free space * permittivity of material
            eps_l = (e0_cm * relative_permittivity_silicon)
            eps_r = (e0_cm * relative_permittivity_silicon)
            eps_u = (e0_cm * relative_permittivity_silicon)
            eps_d = (e0_cm * relative_permittivity_silicon)

            dx_right = eps_r * (epr - epc) / (yr - yc)
            dx_left = eps_l * (epc - epl) / (yc - yl)
            dxx = 2 * (dx_right - dx_left) / (yr - yl)

            dy_up = eps_u * (epu - epc) / (xu - xc)
            dy_down = eps_d * (epc - epd) / (xc - xd)
            dyy = 2 * (dy_up - dy_down) / (xu - xd)
            
            # the final unit of "ds" is in C/(cm * um^2). Multiply by 1e8 to convert the um^2 to cm^2 to get to C/(cm^3).
            div_cm = (dxx + dyy) * 1e8
            
            # Divide by constant Q (charge of electron)
            space_charge = div_cm / -1.60217663e-19
            
            # Final unit of space charge is in 1/(cm^3). This is in the same units as the simulation uses.
            frame_result[i][j] = space_charge
    return frame_result

# Wrapper around albert's function
def derivative_all(data):
    result = torch.zeros_like(data)
    x, y = load_spacing()
    for i in range(101):
        result[i] = derivative_one_frame(data[i], x, y)
    return result

q = 1.60217663e-19

anim = AnimationMaker()
space_charge = torch.tensor(load_space_charge()) * -q
anim.add_data(space_charge, "Space charge")

# Make a cheat about the laplacian
laplacian = derivative_all(torch.tensor(load_elec_potential())) * -q
laplacian[:, 0, :] = space_charge[:, 0, :]
laplacian[:, -1, :] = space_charge[:, -1, :]
laplacian[:, :, 0] = space_charge[:, :, 0]
laplacian[:, :, -1] = space_charge[:, :, -1]
anim.add_data(laplacian, "Laplacian", vmin = np.min(space_charge), vmax = np.max(space_charge))

# dv = DataVisualizer(log_diff(space_charge, laplacian))
# dv.add_data(space_charge, "Space charge")
# dv.add_data(laplacian, "Laplacian")
# dv.show()

anim.add_data(log_diff(space_charge, laplacian), "Log difference", vmin = -8)
anim.add_data(log_diff(space_charge, laplacian), "Log difference", vmin = -5)
anim.add_text([f"Frame {i}" for i in range(101)])
anim.save("derivatives.gif")