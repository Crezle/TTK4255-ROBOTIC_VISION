import numpy as np
from estimate_E import *
from epipolar_distance import *
from F_from_E import *

def estimate_E_ransac(B1, B2, K, distance_threshold, num_trials):

    # Tip: The following snippet extracts a random subset of 8
    # correspondences (w/o replacement) and estimates E using them.
    best_num_inliers = 0
    
    for _ in range(num_trials):
        sample = np.random.choice(B1.shape[1], size=8, replace=False)
        E = estimate_E(B1[:, sample], B2[:, sample])
        e = epipolar_distance(F_from_E(E, K), K @ B1, K @ B2)
        num_inliers = np.sum(e < distance_threshold)
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inliers = e < distance_threshold
    
    E = estimate_E(B1[:, best_inliers], B2[:, best_inliers])
    
    return E
