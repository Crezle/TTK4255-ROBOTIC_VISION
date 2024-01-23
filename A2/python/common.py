import matplotlib.pyplot as plt
import numpy as np

def im2double(im):
    """
    Ensures that the returned image is a floating-point image,
    with pixel values in the range [0,1].
    """
    if im.dtype == np.uint8: return im/255.0
    else: return im

def rgb_to_gray(I):
    """
    Converts a HxWx3 RGB image to a HxW grayscale image by averaging.
    """
    R = I[:,:,0]
    G = I[:,:,1]
    B = I[:,:,2]
    return (R + G + B)/3.0

def derivative_of_gaussian(I, sigma):
    """
    Estimates the horizontal and vertical image derivatives
    by convolving with the derivatives of a 2D Gaussian.
    Returns the derivative images (Ix, Iy) and the gradient
    magnitude Im.
    """
    h = int(np.ceil(3*sigma))
    x = np.arange(2*h + 1) - h
    e = np.exp(-x**2/(2*sigma**2))
    g = e/np.sqrt(2*np.pi*sigma**2)
    d = -x*e/(sigma*sigma*sigma*np.sqrt(2*np.pi))
    Ix = np.zeros_like(I)
    Iy = np.zeros_like(I)
    for row in range(I.shape[0]): Ix[row,:] = np.convolve(I[row,:], d, mode='same')
    for col in range(I.shape[1]): Ix[:,col] = np.convolve(Ix[:,col], g, mode='same')
    for col in range(I.shape[1]): Iy[:,col] = np.convolve(I[:,col], d, mode='same')
    for row in range(I.shape[0]): Iy[row,:] = np.convolve(Iy[row,:], g, mode='same')
    return Ix, Iy, np.sqrt(Ix**2 + Iy**2)

def gaussian(I, sigma):
    """
    Applies a 2-D Gaussian blur with standard deviation sigma to
    a grayscale image I.
    """
    h = int(np.ceil(3*sigma))
    x = np.linspace(-h, +h, 2*h + 1)
    g = np.exp(-x**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
    Ig = np.zeros_like(I)
    for row in range(I.shape[0]): Ig[row,:] = np.convolve(I[row,:], g, mode='same')
    for col in range(I.shape[1]): Ig[:,col] = np.convolve(Ig[:,col], g, mode='same')
    return Ig

def extract_edges(Ix, Iy, Im, threshold):
    """
    Returns the x, y coordinates of pixels whose gradient
    magnitude is greater than the threshold. Also, returns
    the angle of the image gradient at each extracted edge.
    """
    y,x = np.nonzero(Im > threshold)
    theta = np.arctan2(Iy[y,x], Ix[y,x])
    return x, y, theta

def extract_local_maxima(H, threshold):
    """
    Returns the row and column of cells whose value is strictly greater than its
    8 immediate neighbors, and greater than or equal to a threshold. The threshold
    is specified as a fraction of the maximum array value.

    Note: This returns (row,column) coordinates.
    """
    assert(len(H.shape) == 2) # Must be gray-scale array
    assert(threshold >= 0 and threshold <= 1) # Threshold is specified as fraction of maximum array value
    absolute_threshold = threshold*H.max()
    maxima = []
    for row in range(1, H.shape[0]-1):
        for col in range(1, H.shape[1]-1):
            window = H[row-1:row+2, col-1:col+2]
            center = window[1,1]
            window[1,1] = 0.0
            if center > window.max() and center >= absolute_threshold:
                maxima.append((row,col))
            window[1,1] = center
    maxima = np.array(maxima)
    return maxima[:,0], maxima[:,1]

def draw_line(theta, rho, **args):
    """
    Draws a line given in normal form (rho, theta).
    Uses the current plot's xlim and ylim as bounds.
    """
    def clamp(a, b, a_min, a_max, rho, A, B):
        if a < a_min or a > a_max:
            a = np.fmax(a_min, np.fmin(a_max, a))
            b = (rho-a*A)/B
        return a, b

    x_min,x_max = np.sort(plt.xlim())
    y_min,y_max = np.sort(plt.ylim())
    c = np.cos(theta)
    s = np.sin(theta)
    if np.fabs(s) > np.fabs(c):
        x1 = x_min
        x2 = x_max
        y1 = (rho-x1*c)/s
        y2 = (rho-x2*c)/s
        y1,x1 = clamp(y1, x1, y_min, y_max, rho, s, c)
        y2,x2 = clamp(y2, x2, y_min, y_max, rho, s, c)
    else:
        y1 = y_min
        y2 = y_max
        x1 = (rho-y1*s)/c
        x2 = (rho-y2*s)/c
        x1,y1 = clamp(x1, y1, x_min, x_max, rho, c, s)
        x2,y2 = clamp(x2, y2, x_min, x_max, rho, c, s)
    plt.plot([x1, x2], [y1, y2], **args)
