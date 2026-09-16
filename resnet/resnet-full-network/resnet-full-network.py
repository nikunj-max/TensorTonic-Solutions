import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns the network logits as a nested list.
    """
    # Ensure all inputs are NumPy float64 arrays
    x = np.array(x, dtype=np.float64)
    conv1 = np.array(conv1, dtype=np.float64)
    W1_b1 = np.array(W1_b1, dtype=np.float64)
    W2_b1 = np.array(W2_b1, dtype=np.float64)
    W1_b2 = np.array(W1_b2, dtype=np.float64)
    W2_b2 = np.array(W2_b2, dtype=np.float64)
    Ws_b2 = np.array(Ws_b2, dtype=np.float64)
    fc = np.array(fc, dtype=np.float64)
    
    # Helper for ReLU activation
    relu = lambda v: np.maximum(0, v)
    
    # 1. Initial Projection
    out = relu(x @ conv1)
    
    # 2. Identity Block (Block 1)
    shortcut = out
    F_x = relu(out @ W1_b1)
    F_x = F_x @ W2_b1
    out = relu(F_x + shortcut)
    
    # 3. Projection Block (Block 2)
    # The output width changes, so we project the shortcut through Ws_b2
    shortcut = out @ Ws_b2
    F_x = relu(out @ W1_b2)
    F_x = F_x @ W2_b2
    out = relu(F_x + shortcut)
    
    # 4. Final Linear Classifier
    logits = out @ fc
    
    # Round to 4 decimal places and return as a nested list
    return np.round(logits, 4).tolist()