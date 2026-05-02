import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Using np.diag is the most efficient way to transform 
    # a 1D array into a 2D diagonal matrix.
    return np.diag(v)
