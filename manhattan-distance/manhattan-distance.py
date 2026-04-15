import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Convert inputs to numpy arrays to ensure vectorized operations
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    # Calculate the sum of absolute differences: Σ|xi - yi|
    distance = np.sum(np.abs(x_arr - y_arr))
    
    return float(distance)