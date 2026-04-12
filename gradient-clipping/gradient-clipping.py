import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.asarray(g, dtype=float)
    
    # Compute the global L2 norm
    norm = np.linalg.norm(g)
    
    # Handle edge cases: if norm is 0 or max_norm is not positive, return g as is
    if norm == 0 or max_norm <= 0:
        return g.copy()
    
    # Apply clipping rule: if norm exceeds threshold, scale proportionally
    if norm > max_norm:
        return g * (max_norm / norm)
    
    return g.copy()