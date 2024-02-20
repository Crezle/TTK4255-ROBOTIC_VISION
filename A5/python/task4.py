import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import cv2 as cv
import os
np.random.seed(0)

folder = 'A5/data/calibration'

K           = np.loadtxt(join(folder, 'K.txt')) # intrinsic matrix
dc          = np.loadtxt(join(folder, 'dc.txt')) # distortion coefficients
std_int     = np.loadtxt(join(folder, 'std_int.txt')) # standard deviations of intrinsics (entries in K and distortion coefficients)
u_all       = np.load(join(folder, 'u_all.npy')) # detected checkerboard corner locations
image_size  = np.loadtxt(join(folder, 'image_size.txt')).astype(np.int32) # height,width
mean_errors = np.loadtxt(join(folder, 'mean_errors.txt')) # mean error per image

fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy = std_int

dc_std = np.array([k1, k2, p1, p2, k3])

sample_img = cv.imread("A5/data/sample_image.jpg")
h, w = sample_img.shape[:2]

path = "A5/plots/"
if not os.path.exists(path):
    os.makedirs(path)

# Sample distortion coefficients from normal distribution
# Number of samples
n = 5

for i in range(n):
    dc_sampled = np.random.normal(dc, dc_std)
    newcameramtx_sampled, roi_sampled = cv.getOptimalNewCameraMatrix(K, dc, (w, h), 1, (w, h))
    undist_img_samled = cv.undistort(sample_img, K, dc_sampled, None, newcameramtx_sampled)
    cv.imwrite("A5/plots/undistorted_sample_image_sampled_{}.jpg".format(i), undist_img_samled)


newcameramtx, roi = cv.getOptimalNewCameraMatrix(K, dc, (w, h), 1, (w, h))
undist_sample_img = cv.undistort(sample_img, K, dc, None, newcameramtx)

cv.imwrite("A5/plots/undistorted_sample_image_MLE.jpg", undist_sample_img)
cv.imwrite("A5/plots/sample_image.jpg", sample_img)