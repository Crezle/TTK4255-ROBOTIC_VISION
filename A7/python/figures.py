import matplotlib.pyplot as plt
import numpy as np
from common import *

def generate_figure(fig, image_number, K, T, uv, uv_predicted, XY):

    fig.suptitle('Image number %d\nTip: Drag the 3D plot around while holding left click' % image_number)

    #
    # Visualize reprojected markers and estimated object coordinate frame
    #
    I = plt.imread('../data/image%04d.jpg' % image_number)
    plt.subplot(121)
    plt.imshow(I)
    draw_frame(K, T, scale=4.5, labels=False)
    plt.scatter(uv[0,:], uv[1,:], color='red', label='Detected')
    plt.scatter(uv_predicted[0,:], uv_predicted[1,:], marker='+', color='yellow', label='Predicted')
    plt.legend()
    plt.xlim([0, I.shape[1]])
    plt.ylim([I.shape[0], 0])
    plt.axis('off')

    #
    # Visualize scene in 3D
    #
    ax = fig.add_subplot(1,2,2,projection='3d')
    ax.plot(XY[0,:], XY[1,:], np.zeros(XY.shape[1]), '.')
    pO = np.linalg.inv(T)@np.array([0,0,0,1])
    pX = np.linalg.inv(T)@np.array([6,0,0,1])
    pY = np.linalg.inv(T)@np.array([0,6,0,1])
    pZ = np.linalg.inv(T)@np.array([0,0,6,1])
    plt.plot([pO[0], pZ[0]], [pO[1], pZ[1]], [pO[2], pZ[2]], color='#cc4422')
    plt.plot([pO[0], pY[0]], [pO[1], pY[1]], [pO[2], pY[2]], color='#11ff33')
    plt.plot([pO[0], pX[0]], [pO[1], pX[1]], [pO[2], pX[2]], color='#3366ff')
    ax.set_xlim([-40, 40])
    ax.set_ylim([-40, 40])
    ax.set_zlim([-25, 25])
    ax.set_xlabel('x')
    ax.set_zlabel('y')
    ax.set_ylabel('z')

    plt.tight_layout()
