def matrix_factorization_sgd_step(U, V, r, lr, reg):
    """
    Perform one SGD step for matrix factorization.
    """
    # Compute the dot product of U and V
    dot_product = sum(u * v for u, v in zip(U, V))
    
    # Compute the prediction error
    error = r - dot_product
    
    # Update U and V simultaneously using their original values
    U_new = [u + lr * (error * v - reg * u) for u, v in zip(U, V)]
    V_new = [v + lr * (error * u - reg * v) for u, v in zip(U, V)]
    
    return (U_new, V_new)