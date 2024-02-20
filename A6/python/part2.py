import numpy as np
import matplotlib.pyplot as plt
from common import *
from scipy.optimize import least_squares

K = np.loadtxt('../data/K.txt')
u = np.loadtxt('../data/platform_corners_image.txt')
X = np.loadtxt('../data/platform_corners_metric.txt')
I = plt.imread('../data/video0000.jpg')

# This is just an example. Replace these two lines
# with your own code.
hat_T = translate(0.0, 0.0, 1.0)
hat_u = project(K, hat_T@X)

reprojection_errors = np.linalg.norm(u - hat_u, axis=0)
print('Reprojection errors:')
for e in reprojection_errors:
    print('%.05f px' % e)

plt.imshow(I)
plt.scatter(u[0,:], u[1,:], marker='o', facecolors='white', edgecolors='black', label='Detected')
plt.scatter(hat_u[0,:], hat_u[1,:], marker='.', color='red', label='Predicted')
plt.legend()

# Tip: Draw the axes of a coordinate frame
draw_frame(K, hat_T, scale=0.05, labels=True)

# Tip: To zoom in on the platform:
# plt.xlim([200, 500])
# plt.ylim([600, 350])

plt.savefig('out_part2.png')
plt.show()
