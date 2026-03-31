import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    Handles single vectors (3,) and batches (N, 3).
    """
    # Convert input to a NumPy array
    v = np.asanyarray(v, dtype=float)
    
    # Calculate the L2 norm (magnitude)
    # axis=-1 ensures it works for both (3,) and (N, 3) shapes
    # keepdims=True allows for easy broadcasting during division
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    
    # Create a mask for non-zero vectors to avoid division by zero
    # Using a tolerance of 1e-10 as per hints
    mask = norms > 1e-10
    
    # Initialize output as zeros (covers the zero-vector case)
    res = np.zeros_like(v)
    
    # Perform division only where the norm is significant
    # np.where handles the conditional broadcasting efficiently
    res = np.where(mask, v / norms, 0.0)
    
    return res