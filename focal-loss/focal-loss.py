import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Convert inputs to NumPy arrays in case standard Python lists are passed
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Clip probabilities to prevent log(0) or log(1) numerical instability
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1.0 - epsilon)
    
    # Calculate the positive and negative class loss components
    pos_loss = - (1.0 - p) ** gamma * y * np.log(p)
    neg_loss = - p ** gamma * (1.0 - y) * np.log(1.0 - p)
    
    # Return the mean scalar loss across all samples
    return float(np.mean(pos_loss + neg_loss))