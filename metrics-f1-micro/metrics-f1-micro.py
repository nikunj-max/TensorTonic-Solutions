import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Convert inputs to NumPy arrays as suggested
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    
    # Calculate global True Positives
    tp = np.sum(y_t == y_p)
    
    # In single-label multi-class, every error is exactly one FP and one FN globally
    n = len(y_t)
    fp = n - tp
    fn = n - tp
    
    # Handle edge case where there are no true positives to avoid division by zero
    if tp == 0:
        return 0.0
        
    # Micro F1 formula
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn)
    
    return float(f1)