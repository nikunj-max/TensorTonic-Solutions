def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    n = len(X)
    d_in = len(X[0])
    d_out = len(W[0])
    
    Y = []
    
    # Iterate over each sample in the input matrix X
    for i in range(n):
        row_output = []
        # Iterate over each output neuron (column in W)
        for j in range(d_out):
            # Compute the dot product of X's row i and W's column j
            dot_product = sum(X[i][k] * W[k][j] for k in range(d_in))
            # Add the corresponding bias and store it
            row_output.append(dot_product + b[j])
        Y.append(row_output)
        
    return Y