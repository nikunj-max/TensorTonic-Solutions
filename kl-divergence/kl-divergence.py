import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Convert inputs to numpy arrays to ensure vectorized operations
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Clip q with a tiny epsilon to prevent division by zero and log(0)
    q_stable = np.clip(q, eps, None)
    
    # Mask to select elements where p > 0, since 0 * log(0) = 0 by standard convention
    mask = p > 0
    
    # Compute the divergence only for indices where p > 0
    kl_div = np.sum(p[mask] * np.log(p[mask] / q_stable[mask]))
    
    return float(kl_div)