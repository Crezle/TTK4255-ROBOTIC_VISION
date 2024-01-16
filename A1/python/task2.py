import numpy as np
import matplotlib.pyplot as plt
from common import *

### Task 2.1 ###
grass_img = plt.imread("data/grass.jpg")
print(grass_img.shape)

### Task 2.2 ###

grass_img = grass_img / 255 # Scaling from [0, 255] to [0, 1]

fig1 = plt.figure(figsize=(16, 9))

plt.title("Task 2.2, Single Channel Colors")

for i in range(grass_img.shape[2]):
    fig1.add_subplot(2, 2, i+1)
    plt.title(f"Image nr. {i}")
    plt.imshow(grass_img[:,:,i], cmap="Greys_r")

fig1.add_subplot(2, 2, 4)
plt.title("RGB Image")
plt.imshow(grass_img)

### Task 2.3 ###

green_ch = grass_img[:,:,1]
green_ch_thrsh = green_ch > 0.74
fig2 = plt.figure(figsize=(16, 9))

plt.title("Task 2.3, Thresholding Single Channel")

fig2.add_subplot(2, 2, 1)
plt.title("Thresholded Image")
plt.imshow(green_ch_thrsh, cmap="Greys_r")

fig2.add_subplot(2, 2, 2)
plt.title("Green Channel Image")
plt.imshow(green_ch, cmap="Greys_r")

fig2.add_subplot(2, 2, 3)
plt.title("RGB Image")
plt.imshow(grass_img)

### Task 2.4 ### 

fig3 = plt.figure(figsize=(16, 9))

plt.title("Task 2.4, Normalization of RGB Image")

pixel_sum = grass_img[:,:,0] * 0

for i in range(grass_img.shape[2]):
    pixel_sum = pixel_sum + grass_img[:,:,i]

norm_grass_img = grass_img
for i in range(grass_img.shape[2]):
    norm_grass_img[:,:,i] = norm_grass_img[:,:,i] / pixel_sum


plt.imshow(norm_grass_img)

### Task 2.5 ###

fig4 = plt.figure(figsize=(16, 9))

plt.title("Task 2.5, Normalized Thresholding Single Channel")

green_ch = norm_grass_img[:,:,1]
green_ch_thrsh = green_ch > 0.40

fig4.add_subplot(2, 2, 1)
plt.title("Thresholded Image")
plt.imshow(green_ch_thrsh, cmap="Greys_r")

fig4.add_subplot(2, 2, 2)
plt.title("Green Channel Image")
plt.imshow(green_ch, cmap="Greys_r")

fig4.add_subplot(2, 2, 3)
plt.title("RGB Image")
plt.imshow(grass_img)

plt.show()