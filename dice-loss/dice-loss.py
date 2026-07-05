import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Convert inputs to float arrays and flatten to handle 1D and 2D uniformly
    p_flat = np.array(p, dtype=float).flatten()
    y_flat = np.array(y, dtype=float).flatten()
    
    # Compute intersection (element-wise product)
    intersection = np.sum(p_flat * y_flat)
    
    # Compute union components
    sum_p = np.sum(p_flat)
    sum_y = np.sum(y_flat)
    
    # Calculate Dice coefficient with smoothing epsilon
    dice = (2.0 * intersection + eps) / (sum_p + sum_y + eps)
    
    # Return Dice loss
    return 1.0 - dice