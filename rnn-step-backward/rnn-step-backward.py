import numpy as np

def rnn_step_backward(dh, cache):
    """
    Implements the backward pass for a single RNN time step.
    
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
    # Unpack the cache and ensure they are numpy arrays to avoid the 
    # 'unsupported operand type' error seen in image_676991.png
    x_t, h_prev, h_t, W, U, b = [np.array(v) for v in cache] #

    # 1. Tanh Backprop Gate
    # Compute the gradient of the loss with respect to the pre-activation sum (z)
    # dz = dh * (1 - h_t^2)
    dz = dh * (1 - h_t**2) #

    # 2. Parameter Gradients
    # dW is the outer product of the local gradient and the input x_t
    dW = np.outer(dz, x_t) #
    # dU is the outer product of the local gradient and the previous hidden state
    dU = np.outer(dz, h_prev) #
    # db is the sum of gradients (for a single step, it is just dz)
    db = dz #

    # 3. Input and Hidden State Gradients
    # Use matrix multiplication to propagate the gradient back to x_t and h_prev
    dx_t = dz @ W #
    dh_prev = dz @ U #

    return dx_t, dh_prev, dW, dU, db
