import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Let's design a dummy land use field
A = np.reshape([7,2,13,7,2,2], (2,3))
vals = np.unique(A)

# Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
col_dict={1:"blue",
          2:"red",
          13:"orange",
          7:"green"}

# We create a colormar from our list of colors
cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
print(cm(0))
# Let's also define the description of each category : 1 (blue) is Sea; 2 (red) is burnt, etc... Order should be respected here ! Or using another dict maybe could help.
labels = np.array(["Sea","City","Sand","Forest"])
len_lab = len(labels)

# prepare normalizer
## Prepare bins for the normalizer
norm_bins = np.sort([*col_dict.keys()]) #+ 0.5
# norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
print(norm_bins)
## Make normalizer and formatter
norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
# fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

# Plot our figure
# fig,ax = plt.subplots()
# im = ax.imshow(A, cmap=cm, norm=norm)
plt.imshow(A, cmap=cm, norm=norm)
plt.show()
# diff = norm_bins[1:] - norm_bins[:-1]
# tickz = norm_bins[:-1] + diff / 2
# cb = fig.colorbar(im, format=fmt, ticks=tickz)
# fig.savefig("example_landuse.png")
# plt.show()