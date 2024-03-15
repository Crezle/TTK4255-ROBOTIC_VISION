import numpy as np

def epipolar_distance(F, u1, u2):
    """
    F should be the fundamental matrix (use F_from_E).
    u1, u2 should be arrays of size 3 x n containing
    homogeneous pixel coordinates.
    """
    n = u1.shape[1]
    e = np.zeros(n) # Placeholder, replace with your implementation
    return e
