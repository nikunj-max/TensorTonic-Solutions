import math

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    total_loss = 0.0
    n = len(predictions)
    
    for p, y in zip(predictions, targets):
        # 1. Compute the probability assigned to the true class
        p_t = p if y == 1 else 1 - p
        
        # 2. Compute the focal loss for this sample
        sample_loss = -alpha * ((1 - p_t) ** gamma) * math.log(p_t)
        total_loss += sample_loss
        
    # 3. Return the mean focal loss across all samples
    return total_loss / n