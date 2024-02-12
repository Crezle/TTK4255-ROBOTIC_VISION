import numpy as np
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt

#
# Tip: Define functions to create the basic 4x4 transformations
#
# def translate_x(x): Translation along X-axis
# def translate_y(y): Translation along Y-axis
# def translate_z(z): Translation along Z-axis
# def rotate_x(radians): Rotation about X-axis
# def rotate_y(radians): Rotation about Y-axis
# def rotate_z(radians): Rotation about Z-axis
#
# For example:
def translate_x(x):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
#
# Note that you should use np.array, not np.matrix,
# as the latter can cause some unintuitive behavior.
#
# translate_x/y/z could alternatively be combined into
# a single function.

def project(K, X):
    """
    Computes the pinhole projection of a 3xN array of 3D points X
    using the camera intrinsic matrix K. Returns the dehomogenized
    pixel coordinates as an array of size 2xN.
    """
    N = X.shape[1]
    u = np.zeros([2,N])

    u_tilde = K @ X
    u = u_tilde[0:2, :] / u_tilde[2, :]

    return u

def draw_frame(K, T, scale=1, labels=True):
    """
    Visualize the coordinate frame axes of the 4x4 object-to-camera
    matrix T using the 3x3 intrinsic matrix K.

    This uses your project function, so implement it first.

    Control the length of the axes by specifying the scale argument.

    Set labels=False to disable X, Y, Z labels.
    """
    X = T @ np.array([
        [0,scale,0,0],
        [0,0,scale,0],
        [0,0,0,scale],
        [1,1,1,1]])
    u,v = project(K, X)  # If you get an error message here, you should modify your project function to accept 4xN arrays of homogeneous vectors, instead of 3xN.
    plt.plot([u[0], u[1]], [v[0], v[1]], color='#cc4422') # X-axis
    plt.plot([u[0], u[2]], [v[0], v[2]], color='#11ff33') # Y-axis
    plt.plot([u[0], u[3]], [v[0], v[3]], color='#3366ff') # Z-axis
    if labels:
        textargs = {'color': 'w', 'va': 'center', 'ha': 'center', 'fontsize': 'x-small', 'path_effects': [PathEffects.withStroke(linewidth=1.5, foreground='k')]}
        plt.text(u[1], v[1], 'X', **textargs)
        plt.text(u[2], v[2], 'Y', **textargs)
        plt.text(u[3], v[3], 'Z', **textargs)
