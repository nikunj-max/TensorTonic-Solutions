import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the error
    error = y_true - y_pred
    abs_error = np.abs(error)
    
    # Apply the piecewise formula:
    # Quadratic (0.5 * e^2) if |e| <= delta
    # Linear (delta * (|e| - 0.5 * delta)) if |e| > delta
    loss = np.where(
        abs_error <= delta,
        0.5 * np.square(error),
        delta * (abs_error - 0.5 * delta)
    )
    
    # Return the scalar mean loss
    return np.mean(loss)