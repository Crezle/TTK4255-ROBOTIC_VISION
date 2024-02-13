import numpy as np
import matplotlib.pyplot as plt
import os
from common import *

# Create folder for plots if it doesn't exist
path = "A4/plots/"
if not os.path.exists(path):
    os.makedirs(path)

# Load image and data (task 4.2)
quanser_img = plt.imread('A4/data/quanser.jpg')
T_plat_to_cam = np.loadtxt('A4/data/platform_to_camera.txt')
K_cam = np.loadtxt('A4/data/heli_K.txt')

# Define platform points
X_plat = np.array([[0, 0, 0, 1],
                   [0.1145, 0, 0, 1],
                   [0, 0.1145, 0, 1],
                   [0.1145, 0.1145, 0, 1]]).T

# Transform platform points to camera coordinates
X_cam = K_cam @ (T_plat_to_cam @ X_plat)[0:3, :]
X_cam = X_cam / X_cam[2, :]

# Plot image and camera points
plt.imshow(quanser_img)
plt.scatter(X_cam[0, :], X_cam[1, :])
plt.xlim([100, 600])
plt.ylim([600, 300])
plt.savefig('A4/plots/task4-2_platform_to_camera')
if os.getenv("GITHUB_ACTIONS") != 'true':   
    plt.show()
else:
    plt.clf()

# Load image and data (task 4.7)
quanser_img  # Already loaded
T_plat_to_cam  # Already loaded
K_cam  # Already loaded

# Define parameters
psi = np.deg2rad(11.6)
theta = np.deg2rad(28.9)
phi = np.deg2rad(0)

# Define rotations
Rx = np.array([[1, 0, 0],
               [0, np.cos(phi), -np.sin(phi)],
               [0, np.sin(phi), np.cos(phi)]])

Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
               [0, 1, 0],
               [-np.sin(theta), 0, np.cos(theta)]])

Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
               [np.sin(psi), np.cos(psi), 0],
               [0, 0, 1]])

# First three points are in ARM, last four in ROTORS
pts = np.loadtxt('A4/data/heli_points.txt').T

T_rot_to_arm = np.block([[Rx, np.array([0.65, 0, -0.03]).reshape(3, 1)],
                         [0, 0, 0, 1]])

T_arm_to_hinge = np.block([[np.eye(3), np.array([0, 0, -0.05]).reshape(3, 1)],
                           [0, 0, 0, 1]])

T_hinge_to_base = np.block([[Ry, np.array([0, 0, 0.325]).reshape(3, 1)],
                            [0, 0, 0, 1]])

T_base_to_plat = np.block([[Rz, np.array([0.1145/2, 0.1145/2, 0]).reshape(3, 1)],
                           [0, 0, 0, 1]])

T_arm_to_plat = T_base_to_plat @ T_hinge_to_base @ T_arm_to_hinge
T_rot_to_plat = T_arm_to_plat @ T_rot_to_arm

# Transform points to camera coordinates
print(pts.shape)

X_cam_from_arm = K_cam @ (T_plat_to_cam @ T_arm_to_plat @ pts[:, 0:3])[0:3, :]
X_cam_from_arm = X_cam_from_arm / X_cam_from_arm[2, :]

X_cam_from_rot = K_cam @ (T_plat_to_cam @ T_rot_to_plat @ pts[:, 4:])[0:3, :]
X_cam_from_rot = X_cam_from_rot / X_cam_from_rot[2, :]

X_cam = np.concatenate((X_cam_from_arm, X_cam_from_rot), axis=1)

plt.imshow(quanser_img)
plt.scatter(X_cam[0, :], X_cam[1, :], c='yellow')
draw_frame(K_cam, T_plat_to_cam, scale=0.05)

plt.savefig('A4/plots/task4-7_heli_points')
if os.getenv("GITHUB_ACTIONS") != 'true':
    plt.show()
else:
    plt.clf()
