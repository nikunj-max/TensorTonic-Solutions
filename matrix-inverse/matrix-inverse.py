import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Safely convert input to a NumPy array in case a nested list is passed
    A_arr = np.asanyarray(A)
    
    # Validate that the input is a 2D square matrix
    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        return None
        
    try:
        # Compute the inverse using NumPy's optimized LAPACK routines
        return np.linalg.inv(A_arr)
    except np.linalg.LinAlgError:
        # Returns None if the matrix is mathematically singular
        return None