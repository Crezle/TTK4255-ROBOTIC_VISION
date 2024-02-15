import numpy as np
import matplotlib.pyplot as plt
from os.path import join

folder = '../data/calibration'

K           = np.loadtxt(join(folder, 'K.txt'))
dc          = np.loadtxt(join(folder, 'dc.txt'))
std_int     = np.loadtxt(join(folder, 'std_int.txt'))
u_all       = np.load(join(folder, 'u_all.npy'))
image_size  = np.loadtxt(join(folder, 'image_size.txt')).astype(np.int32) # height,width
mean_errors = np.loadtxt(join(folder, 'mean_errors.txt'))

# Extract components of intrinsics standard deviation vector. See:
# https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy = std_int

print()
print('Calibration results')
print('================================')
print('Focal length and principal point')
print('--------------------------------')
print('fx:%13.5f +/- %.5f' % (K[0,0], fx))
print('fy:%13.5f +/- %.5f' % (K[1,1], fy))
print('cx:%13.5f +/- %.5f' % (K[0,2], cx))
print('cy:%13.5f +/- %.5f' % (K[1,2], cy))
print()
print('Distortion coefficients')
print('--------------------------------')
print('k1:%13.5f +/- %.5f' % (dc[0], k1))
print('k2:%13.5f +/- %.5f' % (dc[1], k2))
print('k3:%13.5f +/- %.5f' % (dc[4], k3))
print('p1:%13.5f +/- %.5f' % (dc[2], p1))
print('p2:%13.5f +/- %.5f' % (dc[3], p2))
print('--------------------------------')
print()
print('The number after "+/-" is the standard deviation.')
print()

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.bar(range(len(mean_errors)), mean_errors)
plt.title('Mean error per image')
plt.xlabel('Image index')
plt.ylabel('Mean error (pixels)')
plt.tight_layout()

plt.subplot(122)
for i in range(u_all.shape[0]):
    plt.scatter(u_all[i, :, 0, 0], u_all[i, :, 0, 1], marker='.')
plt.axis('image')
plt.xlim([0, image_size[1]])
plt.ylim([image_size[0], 0])
plt.xlabel('u (pixels)')
plt.ylabel('v (pixels)')
plt.title('All corner detections')
plt.tight_layout()
plt.show()
