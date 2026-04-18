import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Convert input to a numpy array to ensure it has a .shape attribute
    x = np.array(x)
    
    # Initialize the random number generator
    if rng is None:
        rng = np.random.default_rng()

    # Create a mask where elements are kept with probability (1 - p)
    mask = rng.random(x.shape) >= p
    
    # Calculate the scaling factor (Inverted Dropout)
    # If p=1.0, we avoid division by zero, though constraints say p < 1.0
    scale = 1 / (1 - p) if p < 1.0 else 0.0
    
    # Create the pattern: 0 for dropped, 1/(1-p) for kept
    dropout_pattern = mask.astype(float) * scale
    
    # Apply the pattern to the input
    output = x * dropout_pattern
    
    return output, dropout_pattern