import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    v = np.array(v)
    w = np.array(w)
    
    # Calculate the Euclidean norms (magnitudes)
    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)
    
    # Check for zero vectors (threshold 1e-10)
    if norm_v < 1e-10 or norm_w < 1e-10:
        return np.nan
    
    # Compute the dot product
    dot_product = np.dot(v, w)
    
    # Calculate the cosine value
    cos_theta = dot_product / (norm_v * norm_w)
    
    # Clamp the value to the interval [-1, 1] to handle floating point errors
    # This prevents np.arccos from returning NaN for values like 1.0000000000000002
    cos_theta_clamped = np.clip(cos_theta, -1.0, 1.0)
    
    # Return the angle in radians
    return np.arccos(cos_theta_clamped)