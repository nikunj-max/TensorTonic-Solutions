import numpy as np

def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    """
    # Convert to a numpy array and store the original shape
    pts = np.asarray(points)
    orig_shape = pts.shape
    
    # Reshape to a 2D array (N, 3) for vectorized operations
    pts_2d = pts.reshape(-1, 3)
    
    # Compute trigonometric functions once
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Extract x, y, z columns
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]
    z = pts_2d[:, 2]
    
    # Calculate new coordinates
    rotated_pts = np.empty_like(pts_2d, dtype=float)
    rotated_pts[:, 0] = x * cos_t - y * sin_t
    rotated_pts[:, 1] = x * sin_t + y * cos_t
    rotated_pts[:, 2] = z
    
    # Restore and return in the original shape
    return rotated_pts.reshape(orig_shape)