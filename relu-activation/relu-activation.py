import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    Returns the input value if positive, and 0 otherwise.
    Works for scalars, lists, and NumPy arrays.
    """
    # Ensure the input is a NumPy array to handle lists/scalars consistently
    # np.maximum is fully vectorized and works element-wise
    return np.maximum(0, x)