import numpy as np
import matplotlib.pyplot as plt
from common import *

### Task 2.1 ###
grass_img = plt.imread("data/grass.jpg")
print(grass_img.shape)

### Task 2.2 ###
for i in range(grass_img.shape[2]):
    print(i)
    grass_plot = plt.imshow(grass_img[:,:,i], cmap="Greens")
    plt.show()
