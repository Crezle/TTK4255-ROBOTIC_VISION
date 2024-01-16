import numpy as np

def rgb_to_gray(I):
    """
    Converts a HxWx3 RGB image to a HxW grayscale image as
    described in the text.
    """
    R = I[:,:,0]
    G = I[:,:,1]
    B = I[:,:,2]
    return G # Placeholder

def central_difference(I):
    """
    Computes the gradient in the x and y direction using
    a central difference filter, and returns the resulting
    gradient images (Ix, Iy) and the gradient magnitude Im.
    """

    Ix = np.zeros_like(I) # Placeholder
    Iy = np.zeros_like(I) # Placeholder
    Im = np.zeros_like(I) # Placeholder
    return Ix, Iy, Im

def gaussian(I, sigma):
    """
    Applies a 2-D Gaussian blur with standard deviation sigma to
    a grayscale image I.
    """

    # Hint: The size of the kernel should depend on sigma. A common
    # choice is to make the half-width be 3 standard deviations. The
    # total kernel width is then 2*np.ceil(3*sigma) + 1.

    result = np.zeros_like(I) # Placeholder
    return result

def extract_edges(Ix, Iy, Im, threshold):
    """
    Returns the x, y coordinates of pixels whose gradient
    magnitude is greater than the threshold. Also, returns
    the angle of the image gradient at each extracted edge.
    """

    return [0,0,0] # Placeholder
