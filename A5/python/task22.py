import numpy as np

# Calculate the projected pixel coordinates of a single 3x1 point
# in the pinhole camera model with Brown-Conrady distortion.
def project(X, fx, fy, cx, cy, k1, k2, k3, p1, p2):
    x,y = X[:2]/X[2]
    r2 = x**2 + y**2
    r4 = r2*r2
    r6 = r4*r2
    dr = (k1*r2 + k2*r4 + k3*r6)
    dx = dr*x + 2*p1*x*y + p2*(r2 + 2*x*x)
    dy = dr*y + p1*(r2 + 2*y*y) + 2*p2*x*y
    u = cx + fx*(x + dx)
    v = cy + fy*(y + dy)
    return np.array((u, v))

# Maximum likelihood estimates
fx = 2359.40946
fy = 2359.61091
cx = 1370.05852
cy = 1059.63818
k1 = -0.06652
k2 = +0.06534
k3 = -0.07555
p1 = +0.00065
p2 = -0.00419

# Estimated standard deviations
# Multiply these by 1.96 to get the half-width of the 95% confidence interval
std_fx = 0.84200
std_fy = 0.76171
std_cx = 1.25225
std_cy = 0.98041
std_k1 = 0.00109
std_k2 = 0.00624
std_k3 = 0.01126
std_p1 = 0.00011
std_p2 = 0.00014

# Image width and height
W = 2816
H = 2112

# This point will project approximately to the lower-right corner (u=W, v=H)
# when using the maximum likelihood estimates
X = np.array((0.64063963, 0.46381152, 1.0))
print(project(X, fx, fy, cx, cy, k1, k2, k3, p1, p2)) # Check that this is close to (W, H)

