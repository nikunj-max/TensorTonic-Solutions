import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Use len(A) instead of A.shape[0] to handle both lists and arrays
    n = len(A)
    
    # Initialize trace accumulator
    trace_sum = 0
    
    # Iterate through the diagonal indices (i, i)
    for i in range(n):
        trace_sum += A[i][i]
        
    return trace_sum
