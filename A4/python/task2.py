import numpy as np
import matplotlib.pyplot as plt
import os
from common import *

# Create folder for plots
path = "A4/plots/"
if not os.path.exists(path):
    os.makedirs(path)
# Tip: Use np.loadtxt to load data into an array
K = np.loadtxt('A4/data/task2K.txt')
X = np.loadtxt('A4/data/task2points.txt')

# Task 2.2: Implement the project function
u,v = project(K, X)

# You would change these to be the resolution of your image. Here we have
# no image, so we arbitrarily choose a resolution.
width,height = 600,400

#
# Figure for Task 2.2: Show pinhole projection of 3D points
#
plt.figure(figsize=(4,3))
plt.scatter(u, v, c='black', marker='.', s=20)

# The following commands are useful when the figure is meant to simulate
# a camera image. Note: these must be called after all draw commands!

plt.axis('image')     # This option ensures that pixels are square in the figure (preserves aspect ratio)
                      # This must be called BEFORE setting xlim and ylim!
plt.xlim([0, width])
plt.ylim([height, 0]) # The reversed order flips the figure such that the y-axis points down
plt.savefig('A4/plots/task2-2_projection')
if os.getenv("GITHUB_ACTIONS") != 'true':   
    plt.show()
else:
    plt.clf()
