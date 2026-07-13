import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    # Ensure X and y are numpy arrays to allow for array indexing
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Get total number of samples
    N = len(X)
    
    # Create an array of indices and shuffle them
    indices = np.arange(N)
    
    if rng is not None:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)
        
    # Yield batches using the shuffled indices
    for i in range(0, N, batch_size):
        batch_indices = indices[i : i + batch_size]
        
        # Check if we need to drop the last incomplete batch
        if drop_last and len(batch_indices) < batch_size:
            break
            
        yield X[batch_indices], y[batch_indices]