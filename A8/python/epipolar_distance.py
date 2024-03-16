import numpy as np

def epipolar_distance(F, u1, u2):
    """
    F should be the fundamental matrix (use F_from_E).
    u1, u2 should be arrays of size 3 x n containing
    homogeneous pixel coordinates.
    """
    n = u1.shape[1]
    e = np.zeros(n) # Placeholder, replace with your implementation
    
    Fu1 = F @ u1
    FTu2 = F.T @ u2

    n1 = np.sum(u2 * Fu1, axis=0)
    d1 = u1[2, :] * u2[2, :] * np.linalg.norm(Fu1[0:2, :], axis=0)
    
    n2 = np.sum(u1 * FTu2, axis=0)
    d2 = u1[2, :] * u2[2, :] * np.linalg.norm(FTu2[0:2, :], axis=0)
    
    e = np.abs((n1 / d1) + (n2 / d2)) / 2

    return e
