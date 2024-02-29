# Note: You need to install Scipy to run this script. If you don't
# want to install Scipy, then you can look for a different LM
# implementation or write your own.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from quanser import Quanser
from plot_all import *
import os

from common import *

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
    
### Task 3

def get_image_indices(directory, extensions=['.jpg', '.jpeg', '.png', '.gif']):
    files = os.listdir(directory)
    image_files = [file for file in files if any(file.endswith(ext) for ext in extensions)]
    num_images = len(image_files)
    indices = np.linspace(0, num_images-1, 10, dtype=int)  # Get 10 evenly spaced indices
    return indices

# Usage
indices = get_image_indices('A6/quanser_image_sequence/data')

# ModelA

# Get cartesian coordinate of markers in first frame
X_world = np.loadtxt('A6/data/heli_points.txt').T
platform_to_camera = np.loadtxt('A6/data/platform_to_camera.txt')
lengths_A = np.array([0.1145/2, 0.325, -0.050, 0.65, -0.030])


X_world_cartesian = X_world[:-1, :] / X_world[-1, :]

p_A_kinematic = np.concatenate((X_world_cartesian.flatten(), lengths_A)) 
p_A = np.concatenate((p_A_kinematic, all_p[indices, :].flatten()))

# Model B

# Lengths are now initialized for all 9 values (even zeros), "second row" is now a combination
lengths_B = np.array([lengths_A[0], lengths_A[0], 0,
                      lengths_A[2]*np.sin(all_p[0, 1]), 0, lengths_A[1] + lengths_A[2]*np.cos(all_p[0, 1]),
                      lengths_A[3], 0, lengths_A[4]])

angles_B = np.zeros(9)
p_B_kinematic = np.concatenate((X_world_cartesian.flatten(), angles_B, lengths_B))
p_B = np.concatenate((p_B_kinematic, all_p[indices, :].flatten()))

def residuals_A(u, weights, p_A):
    K = np.loadtxt('A6/data/K.txt')
    X = p_A[:21].reshape(3, 7)
    X = np.vstack((X, np.ones(7)))
    lengths = p_A[21:26]
    pose = p_A[26:].reshape(-1, 3)
    hat_u = np.zeros_like(u)
    for i in range(len(pose)):
        base_to_platform = translate(lengths[0], lengths[0], 0.0)@rotate_z(pose[i, 0])
        hinge_to_base    = translate(0.00, 0.00,  lengths[1])@rotate_y(pose[i, 1])
        arm_to_hinge     = translate(0.00, 0.00, lengths[2])
        rotors_to_arm    = translate(lengths[3], 0.00, lengths[4])@rotate_x(pose[i, 2])
        base_to_camera   = platform_to_camera@base_to_platform
        hinge_to_camera  = base_to_camera@hinge_to_base
        arm_to_camera    = hinge_to_camera@arm_to_hinge
        rotors_to_camera = arm_to_camera@rotors_to_arm
        
        p1 = arm_to_camera @ X[:,:3]
        p2 = rotors_to_camera @ X[:,3:]
        
        
        # hat_u is a 2xM array of predicted marker locations.
        hat_u[:, i*7:(i+1)*7] = project(K, np.hstack([p1, p2]))

    r = ((hat_u - u)*weights).flatten()
        
    return r


weights = detections[indices, ::3].flatten()
u = np.concatenate((detections[indices, 1::3].reshape(1, -1), detections[indices, 2::3].reshape(1, -1)), axis=0)

resfun_A = lambda p_A : residuals_A(u, weights, p_A)

p_A = least_squares(resfun_A, x0=p_A, method='lm').x

p = np.array([0.0, 0.0, 0.0])

### Calculate with optimized variables from model A
all_r = []
all_p = []
for i in range(len(detections)):
    weights = detections[i, ::3]
    u = np.vstack((detections[i, 1::3], detections[i, 2::3]))

    # Tip: Lambda functions can be defined inside a for-loop, defining
    # a different function in each iteration. Here we pass in the current
    # image's "u" and "weights".
    resfun = lambda p : quanser.custom_residuals_A(u, weights, lengths=p_A[21:26], heli_points=p_A[0:21], p=p)

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
plot_all(all_p, all_r, detections, subtract_initial_offset=False, task="task32A")
plt.savefig('A6/plots/out_part1b_task32_False.png')
if os.getenv('GITHUB_ACTIONS') != 'true':
    plt.show()
else:
    plt.clf()
    
plot_all(all_p, all_r, detections, subtract_initial_offset=True, task="task32A")
plt.savefig('A6/plots/out_part1b_task32_True.png')
if os.getenv('GITHUB_ACTIONS') != 'true':
    plt.show()
else:
    plt.clf()
