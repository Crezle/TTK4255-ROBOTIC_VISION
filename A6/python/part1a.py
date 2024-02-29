import matplotlib.pyplot as plt
import numpy as np
from gauss_newton import gauss_newton
from quanser import Quanser
import os

if not os.path.exists('A6/plots'):
    os.makedirs('A6/plots')

detections = np.loadtxt('A6/data/detections.txt')
quanser = Quanser()

### Task 1.3 ("a")
image_number = 0

weights = detections[image_number, ::3]
u = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
resfun = lambda p : quanser.residuals(u, weights, p[0], p[1], p[2])

p0 = np.array([11.6, 28.9, 0.0])*np.pi/180

r = resfun(p0)

print('Residuals on image %d:' % image_number)
print(r)

reprojection_errors = np.linalg.norm(r.reshape((2,-1)), axis=0)
print('Reprojection errors:')
for i,reprojection_error in enumerate(reprojection_errors):
    print('Marker %d: %5.02f px' % (i + 1, reprojection_error))
print('Average:  %5.02f px' % np.mean(reprojection_errors))
print('Median:   %5.02f px' % np.median(reprojection_errors))
quanser.draw(u, weights, image_number)
plt.savefig('A6/plots/out_part1a_task13a.png')
if os.getenv("GITHUB_ACTIONS") != 'true':
    plt.show()
else:
    plt.clf()


### Task 1.3 ("b")
image_number = 40

weights = detections[image_number, ::3]
u = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
resfun = lambda p : quanser.residuals(u, weights, p[0], p[1], p[2])

p = gauss_newton(resfun, p0, 0.9, 10)

r = resfun(p)
print('Residuals on image %d:' % image_number)
print(r)

reprojection_errors = np.linalg.norm(r.reshape((2,-1)), axis=0)
reprojection_errors_text = 'Reprojection errors:\n'
for i, reprojection_error in enumerate(reprojection_errors):
    reprojection_errors_text += 'Marker %d: %5.02f px\n' % (i + 1, reprojection_error)
reprojection_errors_text += 'Median:   %5.02f px' % np.median(reprojection_errors)

quanser.draw(u, weights, image_number)
plt.text(0.05, 0.05, reprojection_errors_text, transform=plt.gca().transAxes, va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.savefig('A6/plots/out_part1a_task13b.png')
if os.getenv("GITHUB_ACTIONS") != 'true':
    plt.show()
else:
    plt.clf()

##### Task 1.4 #####

# Test case 1: 
#   p0 equal to first image position
p0 = np.array([11.6, 28.9, 0.0])*np.pi/180
p = gauss_newton(resfun, p0, 0.9, 10, xtol=0.01, print_progress=True)

# Test case 2:
#   p0 equal to an eye approx to position
p0 = np.array([7.0, 5.0, 0.0])*np.pi/180
p = gauss_newton(resfun, p0, 0.9, 10, xtol=0.01, print_progress=True)

# Test case 3:
#   p0 equal to large positional values
p0 = np.array([180.0, 180.0, 180.0])*np.pi/180
p = gauss_newton(resfun, p0, 0.9, 10, xtol=0.01, print_progress=True)


##### Task 1.5 #####
# (Commented out to prevent error)

# image_number = 87

# weights = detections[image_number, ::3]
# u = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
# resfun = lambda p : quanser.residuals(u, weights, p[0], p[1], p[2])
# p0 = np.array([11.6, 28.9, 0.0])*np.pi/180

# p = gauss_newton(resfun, p0, 0.9, 10, xtol=0.01, print_progress=True)
