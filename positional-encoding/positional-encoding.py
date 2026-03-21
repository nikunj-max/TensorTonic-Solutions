import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Initialize the encoding matrix with zeros
    pe = np.zeros((seq_len, d_model))
    
    # Create a column vector for positions: shape (seq_len, 1)
    position = np.arange(seq_len)[:, np.newaxis]
    
    # Create the divisor terms for the dimensions. 
    # We only need it for every 2nd index (0, 2, 4...)
    # The formula uses 2i / d_model
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(base) / d_model))
    
    # Apply sine to even indices (0, 2, 4...)
    pe[:, 0::2] = np.sin(position * div_term)
    
    # Apply cosine to odd indices (1, 3, 5...)
    # If d_model is odd, pe[:, 1::2] will naturally have one less column than div_term,
    # so we slice div_term to match the width of the odd columns.
    pe[:, 1::2] = np.cos(position * div_term[:d_model // 2])
    
    return pe