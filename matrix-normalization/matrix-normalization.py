import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    try:
        # Convert input to numpy array
        matrix = np.array(matrix, dtype=float)
        
        # Validate matrix is 2D
        if matrix.ndim != 2:
            return None

        # Calculate the norm based on type
        if norm_type == 'l1':
            norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        elif norm_type == 'l2':
            norm = np.sqrt(np.sum(np.square(matrix), axis=axis, keepdims=True))
        elif norm_type == 'max':
            norm = np.max(np.abs(matrix), axis=axis, keepdims=True)
        else:
            return None # Invalid norm_type

        # Handle zero vectors to avoid division by zero
        # If norm is 0, we keep it as 1 to avoid NaN (0/1 = 0)
        norm = np.where(norm == 0, 1, norm)

        # Broadcasted division
        normalized_matrix = matrix / norm

        return normalized_matrix

    except Exception:
        return None