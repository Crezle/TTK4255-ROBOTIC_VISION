# Note: You need to install Scipy to run this script. If you don't
# want to install Scipy, then you can look for a different LM
# implementation or write your own.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from quanser import Quanser
from plot_all import *
import os

from enum import Enum

class InitGuess(Enum):
    GOOD = "good_init_guess"
    BAD = "bad_init_guess"

class Model(Enum):
    A = "ModelA"
    B = "ModelB"

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
plot_all(all_p, all_r, detections, subtract_initial_offset=False, task="task17")
plt.savefig('A6/plots/out_part1b_task17_False.png')
if os.getenv('GITHUB_ACTIONS') != 'true':
    plt.show()
else:
    plt.clf()
    
plot_all(all_p, all_r, detections, subtract_initial_offset=True, task="task17")
plt.savefig('A6/plots/out_part1b_task17_True.png')
if os.getenv('GITHUB_ACTIONS') != 'true':
    plt.show()
else:
    plt.clf()
    
##### Task 3 #####

# Get indices of 10 images evenly spaced throughout the sequence
def get_image_indices(directory, extensions=['.jpg', '.jpeg', '.png', '.gif']):
    files = os.listdir(directory)
    image_files = [file for file in files if any(file.endswith(ext) for ext in extensions)]
    num_images = len(image_files)
    indices = np.linspace(0, num_images-1, 10, dtype=int)  # Get 10 evenly spaced indices
    return indices

indices = get_image_indices('A6/quanser_image_sequence/data')

# Load helicopter points
X_world = np.loadtxt('A6/data/heli_points.txt').T
X_world_cartesian = X_world[:-1, :] / X_world[-1, :]

# Initial guess for lengths and angles
for init_guess in InitGuess:

    if init_guess == InitGuess.GOOD:
        lengths_A = np.array([0.1145/2, 0.325, -0.050, 0.65, -0.030])
        lengths_B = np.array([lengths_A[0], lengths_A[0], 0,
                            lengths_A[2]*np.sin(all_p[0, 1]), 0, lengths_A[1] + lengths_A[2]*np.cos(all_p[0, 1]),
                            lengths_A[3], 0, lengths_A[4]])
        angles_B = np.zeros(9)

    elif init_guess == InitGuess.BAD:
        lengths_A = -np.array([0.1145/2, 0.325, -0.050, 0.65, -0.030])
        lengths_B = -np.array([lengths_A[0], lengths_A[0], 0,
                            lengths_A[2]*np.sin(all_p[0, 1]), 0, lengths_A[1] + lengths_A[2]*np.cos(all_p[0, 1]),
                            lengths_A[3], 0, lengths_A[4]])
        angles_B = np.array([np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4])
    else:
        raise ValueError("Init guess not recognized")

    # Create state parameter vectors
    p_A_kinematic = np.concatenate((X_world_cartesian.flatten(), lengths_A)) 
    p_A = np.concatenate((p_A_kinematic, all_p[indices, :].flatten()))
    p_B_kinematic = np.concatenate((X_world_cartesian.flatten(), angles_B, lengths_B))
    p_B = np.concatenate((p_B_kinematic, all_p[indices, :].flatten()))

    # Extract weights and image points
    weights = detections[indices, ::3].flatten()
    u = np.concatenate((detections[indices, 1::3].reshape(1, -1), detections[indices, 2::3].reshape(1, -1)), axis=0)

    # Define residual functions
    resfun_A = lambda p_A : quanser.residuals_model_A(u, weights, p_A)
    p_A = least_squares(resfun_A, x0=p_A, method='lm').x
    resfun_B = lambda p_B : quanser.residuals_model_B(u, weights, p_B)
    p_B = least_squares(resfun_B, x0=p_B, method='lm').x


    ### Calculate pose with optimized variables from both models
    for model in Model:
        p = np.array([0.0, 0.0, 0.0])
        all_r = []
        all_p = []
        for i in range(len(detections)):
            weights = detections[i, ::3]
            u = np.vstack((detections[i, 1::3], detections[i, 2::3]))

            # Tip: Lambda functions can be defined inside a for-loop, defining
            # a different function in each iteration. Here we pass in the current
            # image's "u" and "weights".
            if model == Model.A:
                resfun = lambda p : quanser.residuals_A(u, weights, lengths=p_A[21:26], heli_points=p_A[0:21], p=p)
            elif model == Model.B:
                resfun = lambda p : quanser.residuals_B(u, weights, angles=p_B[21:30], lengths=p_B[30:39], heli_points=p_B[0:21], p=p)
            else:
                raise ValueError("Model not recognized")

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
            
        plot_all(all_p, all_r, detections, subtract_initial_offset=True, task=f"task32A_{model}_{init_guess}")
        plt.savefig(f'A6/plots/out_part3_task32_{model}_{init_guess}.png')
        if os.getenv('GITHUB_ACTIONS') != 'true':
            plt.show()
        else:
            plt.clf()
