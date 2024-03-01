import numpy as np

def estimate_H(xy, XY):
    # Tip: U,s,VT = np.linalg.svd(A) computes the SVD of A.
    # The column of V corresponding to the smallest singular value
    # is the last column, as the singular values are automatically
    # ordered by decreasing magnitude. However, note that it returns
    # V transposed.

    assert XY.shape[1] == xy.shape[1]

    n = XY.shape[1]
    A = np.zeros((2*n, 9))

    x = xy[0, :]
    y = xy[1, :]
    X = XY[0, :]
    Y = XY[1, :]

    # Create A
    for i in range(n):
        A[2*i:2*i+2, 0:9] = np.array([
            [X[i],  Y[i],   1, 0,       0,      0, -X[i]*x[i], -Y[i]*x[i], -x[i]],
            [0,     0,      0, X[i],    Y[i],   1, -X[i]*y[i], -Y[i]*y[i], -y[i]]
        ])

    # Extract V from SVD
    _, _, VT = np.linalg.svd(A)
    V = VT.T

    # Extract h from V and create H
    h = V[:, -1]
    h = h / np.linalg.norm(h)
    H = np.reshape(h, (3, 3))

    return H

def decompose_H(H):
    abs_ks = np.linalg.norm(H, axis=0)[0:2]
    abs_k = np.mean(abs_ks)
    
    pos_rescaled_H = H / abs_k
    neg_rescaled_H = H / -abs_k
    
    pos_r3 = np.cross(pos_rescaled_H[:, 0], pos_rescaled_H[:, 1])
    neg_r3 = np.cross(neg_rescaled_H[:, 0], neg_rescaled_H[:, 1])

    pos_Rmatrx = np.block([pos_rescaled_H[:, 0:2], pos_r3.reshape(3, 1)])
    neg_Rmatrx = np.block([neg_rescaled_H[:, 0:2], neg_r3.reshape(3, 1)])

    s1, pos_Rmatrx = closest_rotation_matrix(pos_Rmatrx)
    s2, neg_Rmatrx = closest_rotation_matrix(neg_Rmatrx)

    # Apply the same scaling to the translation part
    pos_t = pos_rescaled_H[:, 2] / s1[0]
    neg_t = neg_rescaled_H[:, 2] / s2[0]

    T1 = np.block([[pos_Rmatrx, pos_t.reshape(3, 1)],
                   [0, 0, 0, 1]])
    T2 = np.block([[neg_Rmatrx, neg_t.reshape(3, 1)],
                   [0, 0, 0, 1]])
    
    return T1, T2

def closest_rotation_matrix(Q):
    # Ensure Q is a rotation matrix
    U, s, Vt = np.linalg.svd(Q)
    R = U @ Vt
    return s, R
