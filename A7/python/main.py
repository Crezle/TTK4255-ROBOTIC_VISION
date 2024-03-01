import matplotlib.pyplot as plt
import numpy as np
from common import *
from figures import *
from tasks import *

import os
from enum import Enum

K           = np.loadtxt('A7/data/K.txt')
detections  = np.loadtxt('A7/data/detections.txt')
XY          = np.loadtxt('A7/data/XY.txt').T
n_total     = XY.shape[1] # Total number of markers (= 24)

if not os.path.exists('A7/plots'):
    os.makedirs('A7/plots')

class Task(Enum):
    TASK_2  = "task_2"
    TASK_31 = "task_31"
    TASK_32 = "task_32"

# for image_number in range(23): # Use this to run on all images
for task in Task:
    for image_number in [4]: # Use this to run on a single image

        # Load data
        # valid : Boolean mask where valid[i] is True if marker i was detected
        #     n : Number of successfully detected markers (<= n_total)
        #    uv : Pixel coordinates of successfully detected markers
        valid = detections[image_number, 0::3] == True
        uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
        uv = uv[:, valid]
        n = uv.shape[1]

        # Tip: The 'valid' array can be used to perform Boolean array indexing,
        # e.g. to extract the XY values of only those markers that were detected.
        # Use this when calling estimate_H and when computing reprojection error.

        # Tip: Helper arrays with 0 and/or 1 appended can be useful if
        # you want to replace for-loops with array/matrix operations.
        # uv1 = np.vstack((uv, np.ones(n)))
        # XY1 = np.vstack((XY, np.ones(n_total)))
        # XY01 = np.vstack((XY, np.zeros(n_total), np.ones(n_total)))

        uv1 = np.vstack((uv, np.ones(n)))
        XY1 = np.vstack((XY, np.ones(n_total)))

        xy = np.linalg.solve(K, uv1)             # TASK: Compute (x,y) as defined in the text
        H = estimate_H(xy, XY[:, valid])        # TASK: Implement this function
        # uv_from_H = np.zeros((2, n_total))    # TASK: Compute predicted pixel coordinates using H
        X_cam = H @ XY1                          # Compute the 3D points in camera coordinates
        uv_from_H = project(K, X_cam)           # Compute the predicted pixel coordinates

        if task == Task.TASK_2:
            T = np.eye(4)
            fig = plt.figure(figsize=plt.figaspect(0.35))
            plt.clf()
            generate_figure(fig, image_number, K, T, uv, uv_from_H, XY)
            plt.savefig(f'A7/plots/out{image_number}_{task}.png')
            if os.getenv('GITHUB_ACTIONS') != "true":
                plt.show() # Uncomment this to have the figure automatically show up
            plt.clf()
            plt.close()

        elif task == Task.TASK_31:
            T1, T2 = decompose_H(H)
            T = [T1, T2]

            for i in range(len(T)):
                fig = plt.figure(figsize=plt.figaspect(0.35))
                plt.clf()
                generate_figure(fig, image_number, K, T[i], uv, uv_from_H, XY)
                plt.savefig(f'A7/plots/out{image_number}_{task}_matrixT{i+1}.png')
                if os.getenv('GITHUB_ACTIONS') != "true":
                    plt.show()
                plt.clf()
                plt.close()
        # The figure should be saved in the data directory as out0000.png, etc.
        # NB! generate_figure expects the predicted pixel coordinates as 'uv_from_H'.
