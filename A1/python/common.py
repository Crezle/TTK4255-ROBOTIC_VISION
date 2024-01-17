import numpy as np

def rgb_to_gray(I):
    """
    Converts a HxWx3 RGB image to a HxW grayscale image as
    described in the text.
    """
    R = I[:,:,0]
    G = I[:,:,1]
    B = I[:,:,2]
    
    greyscale = (R + G + B) / 3
    
    assert 0 <= np.max(greyscale) <= 255
    assert 0 <= np.min(greyscale) <= 255
    return greyscale

def central_difference(I):
    """
    Computes the gradient in the x and y direction using
    a central difference filter, and returns the resulting
    gradient images (Ix, Iy) and the gradient magnitude Im.
    """
    
    kernel = np.array([0.5, 0, -0.5])
    Ix = np.zeros_like(I)
    Iy = np.zeros_like(I)
    Im = np.zeros_like(I)
    
    for row in range(I.shape[0]):
        Ix[row, :] = np.convolve(I[row, :], kernel, mode='same')
        
    for col in range(I.shape[1]):
        Iy[:, col] = np.convolve(I[:, col], kernel.T, mode='same')
        
    Im = np.sqrt(Ix**2 + Iy**2)
    
    return Ix, Iy, Im # Could also be normalized

def gaussian(I, sigma):
    """
    Applies a 2-D Gaussian blur with standard deviation sigma to
    a grayscale image I.
    """

    # Hint: The size of the kernel should depend on sigma. A common
    # choice is to make the half-width be 3 standard deviations. The
    # total kernel width is then 2*np.ceil(3*sigma) + 1.

    result = np.zeros_like(I) # Placeholder
    kernel = np.arange(-np.ceil(3*sigma), np.ceil(3*sigma) + 1)

    
    for row in range(I.shape[0]):
        result[row, :] = np.convolve(I[row, :], kernel, mode='same')
    
    for col in range(I.shape[1]):
        result[:, col] = np.convolve(result[:, col], kernel, mode='same')
    
    return result

def extract_edges(Ix, Iy, Im, threshold):
    """
    Returns the x, y coordinates of pixels whose gradient
    magnitude is greater than the threshold. Also, returns
    the angle of the image gradient at each extracted edge.
    """

    edge_map = Im > threshold
    
    coords = np.nonzero(edge_map)
    
    theta = np.arctan2(Iy[coords], Ix[coords]).reshape(-1, 1)
    coords = np.transpose(coords)
    x = coords[:, 1]
    y = coords[:, 0]

    return x, y, theta # Placeholder
