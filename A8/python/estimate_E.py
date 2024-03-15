import numpy as np

def estimate_E(B1, B2):
    '''
    args:
    - B1: Array of size 3 x n containing back-projection vectors in image 1.
    - B2: Array of size 3 x n containing back-projection vectors in image 2.
    returns:
    - E: The essential matrix.
    '''
    n = B1.shape[1]

    A = np.empty((n, 9))

    for i in range(n):
        A[i, :] = np.kron(B2[:, i].reshape(1, 3), B1[:, i].reshape(1, 3))

    U, S, Vt = np.linalg.svd(A)
    E = Vt[-1, :].reshape(3, 3)

    U, S, Vt = np.linalg.svd(E)
    S[2] = 0
    E = U @ np.diag(S) @ Vt

    E = E / np.linalg.norm(E)**2 # Satisfying constraint ||E|| = 1

    return E
