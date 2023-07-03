from matplotlib import pyplot as plt
from torchshow.visualization import create_color_map_legacy, create_color_map
import numpy as np

def plot_colormap(cmap):
    n = len(cmap)
    # Calculate number of rows and columns for the grid
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols, nrows))

    # Ensure all axes are the same size and remove their labels
    for i in range(nrows):
        for j in range(ncols):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ax[i][j].set_aspect('equal')

    # Draw squares with colors from the colormap
    for idx, color in enumerate(cmap):
        i = idx // ncols
        j = idx % ncols
        ax[i][j].add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color))

    # Remove unused subplots
    for idx in range(n, nrows*ncols):
        i = idx // ncols
        j = idx % ncols
        fig.delaxes(ax[i][j])

    plt.show()

if __name__ == '__main__':
    cmap = create_color_map_legacy(N=64, normalized=True)
    cmap_new = create_color_map(N=64, normalized=True)
    plot_colormap(cmap)
    plot_colormap(cmap_new)
