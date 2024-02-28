import numpy as np
import matplotlib.pyplot as plt
from common import *
from scipy.optimize import least_squares
import os

K = np.loadtxt('A6/data/K.txt')
u = np.loadtxt('A6/data/platform_corners_image.txt')
X = np.loadtxt('A6/data/platform_corners_metric.txt')
I = plt.imread('A6/quanser_image_sequence/data/video0000.jpg')

# This is just an example. Replace these two lines
# with your own code.

def residuals(u, yaw, pitch, roll, z, y, x):
    
    rot_z = rotate_z(yaw)
    rot_y = rotate_y(pitch)
    rot_x = rotate_x(roll)
    rot = rot_z @ rot_y @ rot_x
    
    hat_T = translate(x, y, z)
    hat_u = project(K, hat_T@rot@X)
    
    r = (hat_u - u).flatten()
    
    return r

resfun = lambda p: residuals(u, p[0], p[1], p[2], p[3], p[4], p[5])

init_guess_roll = np.deg2rad(90)
init_guess_pitch = np.deg2rad(0)
init_guess_yaw = np.deg2rad(5)
init_guess_z = 0.8
init_guess_y = 0.0
init_guess_x = 0.0
p_init = np.array([init_guess_yaw, init_guess_pitch, init_guess_roll, init_guess_z, init_guess_y, init_guess_x])

p_cam = least_squares(resfun, x0=p_init, method='lm').x

hat_T = translate(p_cam[5], p_cam[4], p_cam[3])
hat_R = rotate_z(p_cam[0]) @ rotate_y(p_cam[1]) @ rotate_x(p_cam[2])
hat_u = project(K, hat_T @ hat_R @ X)

reprojection_errors = np.linalg.norm(u - hat_u, axis=0)
print('Reprojection errors:')
for e in reprojection_errors:
    print('%.05f px' % e)

plt.imshow(I)
plt.scatter(u[0,:], u[1,:], marker='o', facecolors='white', edgecolors='black', label='Detected')
plt.scatter(hat_u[0,:], hat_u[1,:], marker='.', color='red', label='Predicted')
plt.legend()

# Tip: Draw the axes of a coordinate frame
draw_frame(K, hat_T, scale=0.05, labels=True)

# Tip: To zoom in on the platform:
plt.xlim([200, 500])
plt.ylim([600, 350])

plt.savefig('A6/plots/out_part2.png')
if os.getenv("GITHUB_ACTIONS") != 'true':
    plt.show()
else:
    plt.clf()
