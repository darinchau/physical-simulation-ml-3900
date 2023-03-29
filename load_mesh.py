import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

data = np.load("mesh_data_electrostatic_potential.npy")

def datainfo():
    print(data.shape)
    print(data.max())
    print(data.min())

def make_anim():
    # Set up figure and axis for animation
    fig, ax = plt.subplots()
    heatmap = ax.imshow(data[0], cmap='hot')

    # Define update function for animation
    def update(frame):
        heatmap.set_data(data[frame])
        return heatmap,

    # Create animation object and display it
    ani = animation.FuncAnimation(fig, update, frames=data.shape[0], interval=50, blit=True)
    plt.show()

make_anim()