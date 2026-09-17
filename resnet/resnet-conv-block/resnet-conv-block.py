import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns the projection residual-block output as a nested list.
    """
    # Convert inputs to NumPy float64 arrays
    x = np.array(x, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    Ws = np.array(Ws, dtype=np.float64)
    
    # First layer of the main path: ReLU(x @ W1)
    out1 = np.dot(x, W1)
    relu1 = np.maximum(0, out1)
    
    # Second layer of the main path (NO ReLU here yet): relu1 @ W2
    main_path = np.dot(relu1, W2)
    
    # Shortcut path: x @ Ws
    shortcut = np.dot(x, Ws)
    
    # Add paths and apply the final ReLU activation
    output = np.maximum(0, main_path + shortcut)
    
    # Round to four decimal places and return as a nested list
    return np.round(output, 4).tolist()