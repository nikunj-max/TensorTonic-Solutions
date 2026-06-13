import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    
    # Apply the trapezoidal rule: sum of 0.5 * (tpr[i] + tpr[i+1]) * (fpr[i+1] - fpr[i])
    area = np.sum(0.5 * (tpr[:-1] + tpr[1:]) * (fpr[1:] - fpr[:-1]))
    
    return float(area)