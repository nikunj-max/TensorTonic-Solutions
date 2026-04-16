def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    # Get input dimensions
    H = len(X)
    W = len(X[0])
    
    # Compute output dimensions (integer division handles discarding partial windows)
    out_h = H // pool_size
    out_w = W // pool_size
    
    # Initialize the output matrix with zeros
    output = [[0 for _ in range(out_w)] for _ in range(out_h)]
    
    for i in range(out_h):
        for j in range(out_w):
            # Define the starting point of the window in the input matrix
            start_row = i * pool_size
            start_col = j * pool_size
            
            # Initialize max_val with the first element of the window
            max_val = X[start_row][start_col]
            
            # Iterate through the pool_size x pool_size window
            for r in range(start_row, start_row + pool_size):
                for c in range(start_col, start_col + pool_size):
                    if X[r][c] > max_val:
                        max_val = X[r][c]
            
            # Assign the maximum value found to the output cell
            output[i][j] = max_val
            
    return output