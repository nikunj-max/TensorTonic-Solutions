import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Ensure inputs are numpy arrays to support advanced indexing
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    n_samples = len(y_true)
    
    # Advanced indexing to pick the probability of the true label for each sample
    # This line causes the "tuple" error if y_pred is a standard Python list
    correct_class_probs = y_pred[np.arange(n_samples), y_true]
    
    # Compute negative log likelihood and return the mean
    return -np.mean(np.log(correct_class_probs))