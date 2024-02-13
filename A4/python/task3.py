import numpy as np
import matplotlib.pyplot as plt
import os
from common import *

path = "A4/plots/"
if not os.path.exists(path):
    os.makedirs(path)

K = np.loadtxt('A4/data/task2K.txt')
X = np.loadtxt('A4/data/task3points.txt')

rho = 15 * np.pi / 180
theta = 45 * np.pi / 180
t = np.array([0, 0, 6]).reshape(3,1)

R_x = np.array([[1, 0, 0],
                [0, np.cos(rho), -np.sin(rho)],
                [0, np.sin(rho), np.cos(rho)]])

R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]])

T = np.block([[R_x @ R_y, t],
              [0, 0, 0, 1]])

X_camera = T @ X
u,v = project(K, X_camera)

width,height = 600,400

plt.figure(figsize=(4,3))
plt.scatter(u, v, c='black', marker='.', s=20)
plt.axis('image')
plt.xlim([0, width])
plt.ylim([height, 0])

draw_frame(K, T, scale=0.5)

plt.savefig('A4/plots/task3-3_projection')

if os.getenv("GITHUB_ACTIONS") != 'true':   
    plt.show()
else:
    plt.clf()
