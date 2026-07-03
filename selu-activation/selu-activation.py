import numpy as np

def selu(x, lam=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    """
    Apply SELU activation element-wise.
    Returns a list of floats rounded to 4 decimal places.
    """
    # Convert input to a numpy array for vectorized operations
    x_arr = np.array(x)
    
    # Apply the SELU formula: lam * x if x > 0, else lam * alpha * (exp(x) - 1)
    result = np.where(x_arr > 0, lam * x_arr, lam * alpha * (np.exp(x_arr) - 1))
    
    # Round to 4 decimal places and convert back to a standard Python list
    return np.round(result, 4).tolist()
