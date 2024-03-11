import numpy as np

def SE3(R,t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def decompose_E(E):
    """
    Computes the four possible decompositions of E into a relative
    pose, as described in Szeliski §11.3.1.

    Returns a list of 4x4 transformation matrices.
    """
    U,_,VT = np.linalg.svd(E)
    R90 = np.array([[0, -1, 0], [+1, 0, 0], [0, 0, 1]])
    R1 = U @ R90 @ VT
    R2 = U @ R90.T @ VT
    if np.linalg.det(R1) < 0: R1 = -R1
    if np.linalg.det(R2) < 0: R2 = -R2
    t1, t2 = U[:,2], -U[:,2]
    return [SE3(R1,t1), SE3(R1,t2), SE3(R2, t1), SE3(R2, t2)]
