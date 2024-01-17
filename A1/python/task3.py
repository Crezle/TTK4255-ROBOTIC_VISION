import numpy as np
import matplotlib.pyplot as plt
from common import *

threshold = 2000 # todo: choose an appropriate value
sigma     = 10 # todo: choose an appropriate value
filename  = '../data/grid.jpg'

I_rgb      = plt.imread(filename)
I_rgb      = I_rgb/255.0
I_gray     = rgb_to_gray(I_rgb)
I_blur     = gaussian(I_gray, sigma)
Ix, Iy, Im = central_difference(I_blur)
x,y,theta  = extract_edges(Ix, Iy, Im, threshold)

fig, axes = plt.subplots(1,5,figsize=[15,4], sharey='row')

plt.suptitle(f'threshold = {threshold}, sigma = {sigma}')

plt.set_cmap('gray')
axes[0].imshow(I_blur)
axes[1].imshow(Ix, vmin=-0.05, vmax=0.05)
axes[2].imshow(Iy, vmin=-0.05, vmax=0.05)
axes[3].imshow(Im, vmin=+0.00, vmax=0.10, interpolation='bilinear')
edges = axes[4].scatter(x, y, s=1, c=theta, cmap='hsv')
fig.colorbar(edges, ax=axes[4], orientation='horizontal', label='$\\theta$ (radians)')
for a in axes:
    a.set_xlim([300, 600])
    a.set_ylim([I_rgb.shape[0], 0])
    a.set_aspect('equal')
axes[0].set_title('Blurred input')
axes[1].set_title('Gradient in x')
axes[2].set_title('Gradient in y')
axes[3].set_title('Gradient magnitude')
axes[4].set_title('Extracted edges')
plt.tight_layout()
# plt.savefig('out_edges.png') # Uncomment to save figure to working directory
plt.show()
