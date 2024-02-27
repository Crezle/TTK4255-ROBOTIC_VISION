# Note: You need to install Scipy to run this script. If you don't
# want to install Scipy, then you can look for a different LM
# implementation or write your own.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from quanser import Quanser
from plot_all import *
import os

if not os.path.exists('A6/plots'):
    os.makedirs('A6/plots')


detections = np.loadtxt('A6/data/detections.txt')
quanser = Quanser()

p = np.array([0.0, 0.0, 0.0])
all_r = []
all_p = []
for i in range(len(detections)):
    weights = detections[i, ::3]
    u = np.vstack((detections[i, 1::3], detections[i, 2::3]))

    # Tip: Lambda functions can be defined inside a for-loop, defining
    # a different function in each iteration. Here we pass in the current
    # image's "u" and "weights".
    resfun = lambda p : quanser.residuals(u, weights, p[0], p[1], p[2])

    # Tip: Use the previous image's parameter estimate as initialization
    p = least_squares(resfun, x0=p, method='lm').x

    # Collect residuals and parameter estimate for plotting later
    all_r.append(resfun(p))
    all_p.append(p)
all_p = np.array(all_p)
all_r = np.array(all_r)

# Tip: This saves the estimated angles to a txt file.
# This can be useful for Part 3.
# np.savetxt('trajectory_from_part1.txt', all_p)
# It can be loaded as
# all_p = np.loadtxt('trajectory_from_part1.txt')

# Tip: See comment in plot_all.py regarding the last argument.
plot_all(all_p, all_r, detections, subtract_initial_offset=False)
plt.savefig('A6/plots/out_part1b_task17_False.png')
if os.getenv('GITHUB_ACTIONS') != 'true':
    plt.show()
else:
    plt.clf()
    
plot_all(all_p, all_r, detections, subtract_initial_offset=True)
plt.savefig('A6/plots/out_part1b_task17_True.png')
if os.getenv('GITHUB_ACTIONS') != 'true':
    plt.show()
else:
    plt.clf()
