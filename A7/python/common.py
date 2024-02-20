import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def project(K, X):
    """
    Computes the pinhole projection of a 3xN array of 3D points X
    using the camera intrinsic matrix K. Returns the dehomogenized
    pixel coordinates as an array of size 2xN.
    """
    uvw = K@X[:3,:]
    uvw /= uvw[2,:]
    return uvw[:2,:]

def draw_frame(K, T, scale=1, labels=False):
    """
    Visualize the coordinate frame axes of the 4x4 object-to-camera
    matrix T using the 3x3 intrinsic matrix K.

    Control the length of the axes by specifying the scale argument.
    """
    X = T @ np.array([
        [0,scale,0,0],
        [0,0,scale,0],
        [0,0,0,scale],
        [1,1,1,1]])
    u,v = project(K, X)
    plt.plot([u[0], u[1]], [v[0], v[1]], color='#cc4422') # X-axis
    plt.plot([u[0], u[2]], [v[0], v[2]], color='#11ff33') # Y-axis
    plt.plot([u[0], u[3]], [v[0], v[3]], color='#3366ff') # Z-axis
    if labels:
        textargs = {'color': 'w', 'va': 'center', 'ha': 'center', 'fontsize': 'x-small', 'path_effects': [PathEffects.withStroke(linewidth=1.5, foreground='k')]}
        plt.text(u[1], v[1], 'X', **textargs)
        plt.text(u[2], v[2], 'Y', **textargs)
        plt.text(u[3], v[3], 'Z', **textargs)
