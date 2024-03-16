import numpy as np

def triangulate_many(u1, u2, P1, P2):
    """
    Arguments
        u: Dehomogenized image coordinates in image 1 and 2 (shape 3, n)
        P: Projection matrix K[R t] for image 1 and 2 (shape 3, 4)
    Returns
        X: Homogeneous coordinates of 3D points (shape 4 x n)
    """
    n = u1.shape[1]

    A = np.empty((4, 4))
    X = np.empty((4, n))

    for i in range(n):
        A = np.block([[(u1[0, i]*P1[2, :] - P1[0, :]).reshape(1, 4)],
                      [(u1[1, i]*P1[2, :] - P1[1, :]).reshape(1, 4)],
                      [(u2[0, i]*P2[2, :] - P2[0, :]).reshape(1, 4)],
                      [(u2[1, i]*P2[2, :] - P2[1, :]).reshape(1, 4)]])
    
        _, _, V = np.linalg.svd(A, full_matrices=True)
        X[:, i] = V[-1, :]
        
    X = X / X[3, :]

    return X
