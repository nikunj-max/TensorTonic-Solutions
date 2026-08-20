import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # Calculate the spectral norm (largest singular value) of the weight matrix
    spectral_norm = float(np.linalg.norm(W_hh, ord=2))
    
    # Generate the gradient norm at each time step
    return [1.0 * (spectral_norm ** t) for t in range(T)]