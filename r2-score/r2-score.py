import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Calculate Total Sum of Squares (SStot)
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    # Handle the constant-target edge case
    if ss_tot == 0.0:
        if np.array_equal(y_true, y_pred):
            return 1.0
        else:
            return 0.0
            
    # Calculate Residual Sum of Squares (SSres)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Calculate and return R² score
    return float(1.0 - (ss_res / ss_tot))