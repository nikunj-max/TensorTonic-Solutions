import math

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    K = len(predictions)
    loss = 0.0
    
    for i in range(K):
        # Determine the smoothed target distribution q_i
        if i == target:
            q_i = (1 - epsilon) + (epsilon / K)
        else:
            q_i = epsilon / K
            
        # Accumulate the cross-entropy loss
        loss -= q_i * math.log(predictions[i])
        
    return loss