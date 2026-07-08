import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    losses = []
    
    for y, p in zip(y_true, y_pred):
        # Clip the predicted probability to prevent log(0)
        p_clipped = max(eps, min(1 - eps, p))
        
        # Calculate the log loss using the binary cross-entropy formula
        loss = -(y * math.log(p_clipped) + (1 - y) * math.log(1 - p_clipped))
        losses.append(loss)
        
    return losses