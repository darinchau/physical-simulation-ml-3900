import numpy as np
import matplotlib.pyplot as plt
from load import load_elec_potential, load_spacing

# Load the data
data = load_elec_potential()

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

    # Get the values of the cell across all arrays
    values = data[:, col, row]

    # Plot the values
    plt.figure()
    plt.plot(values)
    plt.xlabel("Array index")
    plt.ylabel("Value")
    plt.title("Values for cell ({}, {})".format(row, col))
    plt.show()

# Create the heatmap
fig, ax = plt.subplots()

ax.set_aspect("equal", adjustable="box")
ax.set_ylabel("Y", va="bottom")
ax.set_yticks([])
heatmap = ax.pcolormesh(x_grid, y_grid, np.transpose(data[0]), cmap="hot", vmin = np.min(data), vmax = np.max(data))

cbar = fig.colorbar(heatmap, ax=ax)
cbar.ax.set_ylabel("Intensity", rotation=-90, va="bottom")
tk = np.round(np.linspace(np.min(data), np.max(data), 4, endpoint=True), 2)
cbar.set_ticks(tk)

# Add a title
fig.suptitle("Visualize data")

# Add the click event handler
cid = fig.canvas.mpl_connect("button_press_event", lambda event: onclick(event, ax))

# Show the plot
plt.show()