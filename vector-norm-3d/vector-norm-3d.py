import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    # Convert input to a numpy array
    v = np.asarray(v, dtype=float)
    
    # Check if the input is a single vector or a batch
    if v.ndim == 1:
        # Single vector: sum of squares along the single axis
        return float(np.sqrt(np.sum(v**2)))
    else:
        # Batch of vectors: sum of squares along axis 1 (the coordinates)
        return np.sqrt(np.sum(v**2, axis=1))