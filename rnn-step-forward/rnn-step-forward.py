import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Computes the forward pass for a single time step of a vanilla RNN.
    
    Arguments:
    x_t    -- Input data at time step t, numpy array of shape (D,)
    h_prev -- Previous hidden state, numpy array of shape (H,)
    Wx     -- Weight matrix for input-to-hidden, shape (D, H)
    Wh     -- Weight matrix for hidden-to-hidden, shape (H, H)
    b      -- Bias vector, shape (H,)

    Returns: 
    h_t    -- New hidden state, shape (H,)
    """
    # Compute the pre-activation (linear combination)
    # x_t @ Wx yields shape (H,), h_prev @ Wh yields shape (H,)
    pre_act = x_t @ Wx + h_prev @ Wh + b
    
    # Apply the tanh activation function
    h_t = np.tanh(pre_act)
    
    return h_t
