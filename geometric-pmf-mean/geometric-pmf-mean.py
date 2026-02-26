import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    # Ensure k is a numpy array for vectorized math
    k_arr = np.asarray(k)
    
    # Calculate PMF: (1-p)^(k-1) * p
    # This works element-wise on the array
    pmf = np.power(1 - p, k_arr - 1) * p
    
    # Calculate Mean: 1/p
    mean = 1.0 / p
    
    return pmf, float(mean)