import numpy as np
import matplotlib.pyplot as plt
import os
from common import *

path = "A4/plots/"
if not os.path.exists(path):
    os.makedirs(path)

K = np.loadtxt('A4/data/task2K.txt')
X = np.loadtxt('A4/data/task2points.txt')

u, v = project(K, X)

width, height = 600, 400

plt.figure(figsize=(4, 3))
plt.scatter(u, v, c='black', marker='.', s=20)

plt.axis('image')
plt.xlim([0, width])
plt.ylim([height, 0])
plt.savefig('A4/plots/task2-2_projection')

if os.getenv("GITHUB_ACTIONS") != 'true':
    plt.show()
else:
    plt.clf()
