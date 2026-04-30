import numpy as np

def calculate_eigenvalues(matrix):
    try:
        # If the input is jagged, np.array/asarray might raise the ValueError 
        # you're seeing in newer NumPy versions.
        matrix = np.array(matrix, dtype=float)
    except (ValueError, TypeError):
        return None

    # Ensure it's 2D and square
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
        
    try:
        # Calculate eigenvalues using NumPy's linalg module
        eigenvalues = np.linalg.eigvals(matrix)
        
        # Sort eigenvalues: primarily by real part, secondarily by imaginary part
        # np.lexsort handles keys from last to first, so we pass (imag, real)
        sort_indices = np.lexsort((eigenvalues.imag, eigenvalues.real))
        sorted_eigenvalues = eigenvalues[sort_indices]
        
        return sorted_eigenvalues
        
    except np.linalg.LinAlgError:
        # Handle cases where computation does not converge
        return None