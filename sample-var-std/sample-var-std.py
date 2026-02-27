import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation using Bessel's correction.
    
    Args:
        x (list or np.array): Numeric data with n >= 2
        
    Returns:
        tuple: (variance, standard_deviation) as scalar floats
    """
    # Convert input to a numpy array for vectorized operations
    x = np.array(x)
    n = len(x)
    
    # Calculate the sample mean
    mean_x = np.mean(x)
    
    # Calculate variance using Bessel's correction (n - 1)
    # Equation: s^2 = sum((xi - mean)^2) / (n - 1)
    variance = np.sum((x - mean_x) ** 2) / (n - 1)
    
    # Calculate standard deviation
    # Equation: s = sqrt(s^2)
    std_dev = np.sqrt(variance)
    
    return float(variance), float(std_dev)