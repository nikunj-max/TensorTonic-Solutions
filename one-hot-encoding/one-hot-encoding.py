import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    # Ensure y is a numpy array
    y = np.array(y)
    
    # Determine the number of classes (K)
    if num_classes is None:
        num_classes = np.max(y) + 1
        
    # Validate that all labels are within the allowed range [0, num_classes - 1]
    if np.any(y >= num_classes) or np.any(y < 0):
        raise ValueError("Labels must be non-negative and less than num_classes.")
    
    # Get the number of samples (N)
    N = y.shape[0]
    
    # Initialize a matrix of zeros of shape (N, K)
    one_hot_matrix = np.zeros((N, num_classes), dtype=float)
    
    # Use advanced indexing to set the appropriate indices to 1
    # np.arange(N) creates row indices [0, 1, ..., N-1]
    # y provides the column indices
    one_hot_matrix[np.arange(N), y] = 1.0
    
    return one_hot_matrix