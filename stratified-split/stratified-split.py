import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Initialize a boolean mask for the test set
    test_mask = np.zeros(len(X), dtype=bool)
    
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        
        if rng is not None:
            rng.shuffle(idx)
        else:
            np.random.shuffle(idx)
            
        n_class = len(idx)
        n_test = int(np.round(n_class * test_size))
        
        # Ensure at least one sample remains in train if possible
        if n_test == n_class and n_class > 1:
            n_test = n_class - 1
        elif n_class == 1:
            n_test = 0
            
        # Flip the selected test indices to True
        test_mask[idx[:n_test]] = True
        
    # Train mask is the logical inverse of the test mask
    train_mask = ~test_mask
    
    # Masking automatically preserves the original relative order!
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]