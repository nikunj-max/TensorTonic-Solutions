import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    y = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)
    
    # If one side is empty, the split provides no information
    if not np.any(split_mask) or np.all(split_mask):
        return 0.0
        
    # Partition the labels
    y_left = y[split_mask]
    y_right = y[~split_mask]
    
    # Calculate sizes
    n = len(y)
    n_left = len(y_left)
    n_right = len(y_right)
    
    # Calculate entropy for parent and children
    h_parent = _entropy(y)
    h_left = _entropy(y_left)
    h_right = _entropy(y_right)
    
    # Calculate information gain
    ig = h_parent - ((n_left / n) * h_left + (n_right / n) * h_right)
    
    return float(ig)
