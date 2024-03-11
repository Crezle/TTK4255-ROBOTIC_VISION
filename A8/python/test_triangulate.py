import numpy as np
from triangulate_many import *

# Generate some random 3D points between [-5,+5] in the world frame.
# The coordinates are rounded to integers to make it easier to read.
num_points = 3
np.random.seed(1)
X = np.rint(-5.0 + 10.0*np.random.ranf(size=(4,num_points)))
X[3,:] = 1

# Make up some projection matrices.
P1 = np.array([[ 0.9211, 0.0000,-0.3894, 0.0000],
               [ 0.0000, 1.0000, 0.0000, 0.0000],
               [ 0.3894, 0.0000, 0.9211, 6.0000]])
P2 = np.array([[ 0.9211, 0.0000, 0.3894, 0.0000],
               [ 0.0000, 1.0000, 0.0000, 0.0000],
               [-0.3894, 0.0000, 0.9211, 6.0000]])

# Perspective projection.
u1 = P1@X
u2 = P2@X
u1 /= u1[2]
u2 /= u2[2]

X_hat = triangulate_many(u1, u2, P1, P2)

print('True vs. estimated 3D coordinates')
print('---------------------------------')
for i in range(num_points):
    print('True:', X[:,i])
    print('Est.:', X_hat[:,i])

if X_hat.shape[0] != 4:
    print('Triangulation is NOT GOOD. The coordinates should be homogeneous.')
elif np.any(np.linalg.norm(X - X_hat, axis=0) > 1e-10):
    print('Triangulation is NOT GOOD. Estimated points do not match the true points.')
else:
    print('Triangulation is GOOD.')
