import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns the bottleneck residual-block output as a nested list.
    """
    # Convert inputs to NumPy float64 arrays
    x = np.array(x, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    W3 = np.array(W3, dtype=np.float64)
    
    # First layer (Width reduction): ReLU(x @ W1)
    out1 = np.maximum(0, np.dot(x, W1))
    
    # Second layer (Bottleneck processing): ReLU(out1 @ W2)
    out2 = np.maximum(0, np.dot(out1, W2))
    
    # Third layer (Width expansion): out2 @ W3 (No ReLU here)
    main_path = np.dot(out2, W3)
    
    # Shortcut path
    if Ws is None:
        shortcut = x
    else:
        Ws = np.array(Ws, dtype=np.float64)
        shortcut = np.dot(x, Ws)
        
    # Add paths and apply the final ReLU activation (as specified in Requirements/Hint 3)
    output = np.maximum(0, main_path + shortcut)
    
    # Round to four decimal places and return as a nested list
    return np.round(output, 4).tolist()