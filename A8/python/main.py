import matplotlib.pyplot as plt
import numpy as np
from figures import *
from estimate_E import *
from decompose_E import *
from triangulate_many import *
from epipolar_distance import *
from F_from_E import *

# Custom imports
from estimate_E_ransac import *
import os
from enum import Enum

class Tasks(Enum):
    Task2 = 2
    Task3 = 3
    Task4 = 4

if not os.path.exists('A8/plots'):
    os.makedirs('A8/plots')

K = np.loadtxt('A8/data/K.txt')
I1 = plt.imread('A8/data/image1.jpg')/255.0
I2 = plt.imread('A8/data/image2.jpg')/255.0
matches = np.loadtxt('A8/data/matches.txt')

u1 = np.vstack([matches[:,:2].T, np.ones(matches.shape[0])])
u2 = np.vstack([matches[:,2:4].T, np.ones(matches.shape[0])])

for task in Tasks:

    if task == Tasks.Task4:
        matches = np.loadtxt('A8/data/task4matches.txt') # Part 4

    if task == Tasks.Task2 or task == Tasks.Task4:
        u1 = np.vstack([matches[:,:2].T, np.ones(matches.shape[0])])
        u2 = np.vstack([matches[:,2:4].T, np.ones(matches.shape[0])])

        B1 = np.linalg.solve(K, u1)
        B2 = np.linalg.solve(K, u2)
        
        if task == Tasks.Task2:
            E = estimate_E(B1, B2)
        elif task == Tasks.Task4:
            #E = estimate_E_ransac(B1, B2)
            continue
        
        # np.random.seed(123) # Leave as commented out to get a random selection each time
        draw_correspondences(I1, I2, u1, u2, F_from_E(E, K), sample_size=8)

    elif task == Tasks.Task3:
        Ts = decompose_E(E)
        T = None
        for i in range(len(Ts)):
            P1 = K @ np.block([np.eye(3), np.zeros((3,1))]) @ np.block([[np.eye(3), np.zeros((3,1))], 
                                                                        [np.array([0, 0, 0, 1])]])
            P2 = K @ np.block([np.eye(3), np.zeros((3,1))]) @ Ts[i]
            X = triangulate_many(u1, u2, P1, P2)
            if np.min(X[2, :]) > 0 and np.min((Ts[i] @ X)[2, :]) > 0:
                T = Ts[i]
                break
            
        if T is None:
            raise ValueError('No valid pose found')

        draw_point_cloud(X, I1, u1,
                         xlim=[np.min(X[0, :]) - 1, np.max(X[0, :]) + 1],
                         ylim=[np.min(X[1, :]) - 1, np.max(X[1, :]) + 1],
                         zlim=[np.min(X[2, :]) * .1, np.max(X[2, :]) + 1])
    else:
        raise ValueError('Invalid task')

    plt.savefig(f'A8/plots/task{task.value}.png')
    if os.getenv('GITHUB_ACTIONS') != "true":
        plt.show()
    else:
        plt.clf()
