import numpy as np

def triangulate_many(u1, u2, P1, P2):
    """
    Arguments
        u: Dehomogenized image coordinates in image 1 and 2 (shape 3, n)
        P: Projection matrix K[R t] for image 1 and 2 (shape 3, 4)
    Returns
        X: Dehomogenized coordinates of 3D points (4 x k)
        where k = n - number of points that could not be triangulated
        u: Dehomogenized image coordinates in image 1 and 2 (shape 3, k)
    """
    n = u1.shape[1]

    A = np.empty((4, 4))
    X = np.zeros((4, n))
    errors = np.zeros(n)

    for i in range(n):
        A = np.block([[(u1[0, i]*P1[2, :] - P1[0, :]).reshape(1, 4)],
                      [(u1[1, i]*P1[2, :] - P1[1, :]).reshape(1, 4)],
                      [(u2[0, i]*P2[2, :] - P2[0, :]).reshape(1, 4)],
                      [(u2[1, i]*P2[2, :] - P2[1, :]).reshape(1, 4)]])
    
        _, _, Vt = np.linalg.svd(A, full_matrices=True)
        X[:, i] = Vt[-1, :]
        errors[i] = np.linalg.norm(A @ X[:, i])

    errors = (errors - np.mean(errors)) / np.std(errors)
    if np.any(errors > 2):
        mask = errors <= 2
        X = X[:, mask]
        u1 = u1[:, mask]
        u2 = u2[:, mask]
        print(f"Could not triangulate {np.sum(~mask)} points")

    X = X / X[3, :]

    return X, u1, u2
