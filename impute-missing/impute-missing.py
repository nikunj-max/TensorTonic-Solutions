import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    # Convert to float and create a copy to avoid modifying the original array
    X_imputed = np.array(X, dtype=float, copy=True)
    
    # Handle 1D arrays by temporarily reshaping to 2D
    is_1d = (X_imputed.ndim == 1)
    if is_1d:
        X_imputed = X_imputed.reshape(-1, 1)
        
    # Iterate over each column
    for col_idx in range(X_imputed.shape[1]):
        col_data = X_imputed[:, col_idx]
        nan_mask = np.isnan(col_data)
        
        # If the column is entirely NaN, fill with 0
        if np.all(nan_mask):
            X_imputed[:, col_idx] = 0.0
        # If the column has some NaNs, impute using the chosen strategy
        elif np.any(nan_mask):
            valid_values = col_data[~nan_mask]
            
            if strategy == 'mean':
                stat_val = np.mean(valid_values)
            elif strategy == 'median':
                stat_val = np.median(valid_values)
            else:
                raise ValueError("Strategy must be 'mean' or 'median'")
                
            X_imputed[nan_mask, col_idx] = stat_val
            
    # Revert back to 1D if the original input was 1D
    if is_1d:
        X_imputed = X_imputed.ravel()
        
    return X_imputed