import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    
    if n_total == 0:
        return 0.0
        
    def _node_gini(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)
        
    gini_left = _node_gini(y_left)
    gini_right = _node_gini(y_right)
    
    weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
    
    return float(weighted_gini)