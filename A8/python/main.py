import matplotlib.pyplot as plt
import numpy as np
from figures import *
from estimate_E import *
from decompose_E import *
from triangulate_many import *
from epipolar_distance import *
from F_from_E import *

# Custom imports
import os
from enum import Enum

if not os.path.exists('A8/plots'):
    os.makedirs('A8/plots')

K = np.loadtxt('A8/data/K.txt')
I1 = plt.imread('A8/data/image1.jpg')/255.0
I2 = plt.imread('A8/data/image2.jpg')/255.0
matches = np.loadtxt('A8/data/matches.txt')
# matches = np.loadtxt('../data/task4matches.txt') # Part 4

class Tasks(Enum):
    Task2 = 2
    Task3 = 3

for task in Tasks:
    u1 = np.vstack([matches[:,:2].T, np.ones(matches.shape[0])])
    u2 = np.vstack([matches[:,2:4].T, np.ones(matches.shape[0])])

    # Task 2: Estimate E
    B1 = np.linalg.solve(K, u1)
    B2 = np.linalg.solve(K, u2)
    E = estimate_E(B1, B2)

    # Task 3: Triangulate 3D points
    if task == Tasks.Task3:
        Ts = decompose_E(E)
        T = Ts[0] # FIX THIS
        P1 = K @ np.block([np.eye(3), np.zeros((3,1))]) @ T
        X = triangulate_many(u1, u2, np.eye(3, 4), np.eye(3, 4))

    #
    # Uncomment in Task 2
    #
    # np.random.seed(123) # Leave as commented out to get a random selection each time
    draw_correspondences(I1, I2, u1, u2, F_from_E(E, K), sample_size=8)

    #
    # Uncomment in Task 3
    #
    if task == Tasks.Task3:
        draw_point_cloud(X, I1, u1, xlim=[-1,+1], ylim=[-1,+1], zlim=[1,3])
    
    plt.savefig(f'A8/plots/task{task.value}.png')
    if os.getenv('GITHUB_ACTIONS') != "true":
        plt.show()
    else:
        plt.clf()
