import numpy as np
import matplotlib.pyplot as plt
import os
from common import *

# Create folder for plots if it doesn't exist
path = "A4/plots/"
if not os.path.exists(path):
    os.makedirs(path)

# Load image and data
quanser_img = plt.imread('A4/data/quanser.jpg')
T_platform = np.loadtxt('A4/data/platform_to_camera.txt')
K_camera = np.loadtxt('A4/data/heli_K.txt')

# Define platform points
X_platform = np.array([[0, 0, 0, 1],
                       [0.1145, 0, 0, 1],
                       [0, 0.1145, 0, 1],
                       [0.1145, 0.1145, 0, 1]]).T

# Transform platform points to camera coordinates
X_camera = K_camera @ (T_platform @ X_platform)[0:3, :]
X_camera = X_camera / X_camera[2, :]

# Plot image and camera points
plt.imshow(quanser_img)
plt.scatter(X_camera[0, :], X_camera[1, :])
plt.xlim([100, 600])
plt.ylim([600, 300])
plt.savefig('A4/plots/task4-2_platform_to_camera')
if os.getenv("GITHUB_ACTIONS") != 'true':   
    plt.show()
else:
    plt.clf()
