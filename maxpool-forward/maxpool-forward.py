def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    H = len(X)
    W = len(X[0])
    
    # Compute output dimensions
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    # Initialize the output matrix
    out = [[0] * W_out for _ in range(H_out)]
    
    # Slide the pooling window over the input
    for i in range(H_out):
        for j in range(W_out):
            row_start = i * stride
            col_start = j * stride
            
            # Find the maximum value within the current window
            max_val = float('-inf')
            for a in range(pool_size):
                for b in range(pool_size):
                    val = X[row_start + a][col_start + b]
                    if val > max_val:
                        max_val = val
                        
            out[i][j] = max_val
            
    return out