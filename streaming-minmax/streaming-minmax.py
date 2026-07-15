import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    return {
        'min': np.full(D, np.inf),
        'max': np.full(D, -np.inf)
    }

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    X_batch = np.asarray(X_batch, dtype=float)
    
    # Calculate current batch min and max
    batch_min = np.min(X_batch, axis=0)
    batch_max = np.max(X_batch, axis=0)
    
    # Update global running min and max
    state['min'] = np.minimum(state['min'], batch_min)
    state['max'] = np.maximum(state['max'], batch_max)
    
    # Normalize the incoming batch with the updated global stats
    range_val = state['max'] - state['min']
    X_norm = (X_batch - state['min']) / (range_val + eps)
    
    return X_norm