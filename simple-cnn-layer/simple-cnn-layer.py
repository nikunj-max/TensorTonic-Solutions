import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # Ensure inputs are numpy arrays to avoid 'attribute shape' errors
    x = np.asarray(x)
    W = np.asarray(W)
    b = np.asarray(b)

    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = W.shape
    
    # Calculate output dimensions: H_out = H - KH + 1
    H_out = H - KH + 1
    W_out = W_in - KW + 1
    
    # Initialize output tensor with float64 precision
    y = np.zeros((N, C_out, H_out, W_out), dtype=np.float64)
    
    # Iterate through batches and output channels
    for n in range(N):
        for cout in range(C_out):
            # Start with the bias for the specific output channel
            y[n, cout] = b[cout]
            
            # Sum contributions from all input channels
            for cin in range(C_in):
                # Optimized sliding window using kernel-index loops and NumPy slicing
                for i in range(KH):
                    for j in range(KW):
                        # Multiply specific weight by the shifted input slice
                        y[n, cout] += x[n, cin, i : i + H_out, j : j + W_out] * W[cout, cin, i, j]
                        
    return y