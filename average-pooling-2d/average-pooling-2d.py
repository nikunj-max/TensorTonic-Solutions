def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    # Get input dimensions
    H = len(X)
    W = len(X[0]) if H > 0 else 0
    
    # Compute output dimensions using integer division (discarding remainders)
    H_out = H // pool_size
    W_out = W // pool_size
    
    # Initialize the output matrix
    output = []
    
    # Total number of elements in each pooling window
    window_area = pool_size * pool_size
    
    # Iterate through each position in the output matrix
    for i in range(H_out):
        output_row = []
        for j in range(W_out):
            # Calculate the top-left starting point of the current window
            start_r = i * pool_size
            start_c = j * pool_size
            
            # Sum up all elements within the pool_size x pool_size window
            window_sum = 0.0
            for a in range(pool_size):
                for b in range(pool_size):
                    window_sum += X[start_r + a][start_c + b]
            
            # Compute the arithmetic mean and append to the current row
            output_row.append(window_sum / window_area)
            
        output.append(output_row)
        
    return output