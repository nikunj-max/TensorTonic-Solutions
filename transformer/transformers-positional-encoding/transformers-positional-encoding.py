import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix of shape (seq_length, d_model).
    """
    # Initialize the encoding matrix with float64
    pe = np.zeros((seq_length, d_model), dtype=np.float64)
    
    # Hint 1: Create positions with shape (seq_length, 1)
    pos = np.arange(seq_length, dtype=np.float64)[:, np.newaxis]
    
    # Hint 2: Compute the inverse frequency for each even column
    # i_2 represents 2i: [0, 2, 4, ..., d_model - 2]
    i_2 = np.arange(0, d_model, 2, dtype=np.float64)
    
    # Base 10000 and the specified exponent (2i / d_model)
    inv_freq = 1.0 / (10000.0 ** (i_2 / d_model))
    
    # Calculate the angles by multiplying positions and inverse frequencies
    angles = pos * inv_freq
    
    # Hint 3: Assign sine to even indices (0::2) and cosine to odd indices (1::2)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    
    return pe