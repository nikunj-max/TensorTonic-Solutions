import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # Sort scores in descending order. 
    # To handle ties properly, we use stable sorting or rely on indexing.
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    # Identify distinct thresholds.
    # We find where the score changes to ensure tied scores are grouped together.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    # The last element is always a threshold point
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    # Calculate cumulative True Positives (TP) and False Positives (FP)
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    # Total count of positives and negatives
    total_pos = tps[-1]
    total_neg = fps[-1]
    
    # Prevent division by zero if there are no positive or negative samples
    tpr = tps / total_pos if total_pos > 0 else np.ones_like(tps)
    fpr = fps / total_neg if total_neg > 0 else np.ones_like(fps)
    
    # Add the initial point (FPR = 0, TPR = 0, threshold = inf)
    fpr = np.r_[0.0, fpr]
    tpr = np.r_[0.0, tpr]
    thresholds = np.r_[np.inf, y_score[threshold_idxs]]
    
    return fpr.tolist(), tpr.tolist(), thresholds.tolist()