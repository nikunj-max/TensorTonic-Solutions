import numpy as np

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    X = np.array(X)
    y = np.array(y)
    
    def calculate_gini(labels):
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return 1.0 - np.sum(probs ** 2)
    
    parent_gini = calculate_gini(y)
    best_gain = -1.0
    best_feature = -1
    best_threshold = -1.0
    
    n_samples, n_features = X.shape
    
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        unique_vals = np.sort(np.unique(feature_values))
        
        if len(unique_vals) <= 1:
            continue
            
        for i in range(len(unique_vals) - 1):
            threshold = (unique_vals[i] + unique_vals[i + 1]) / 2.0
            
            left_mask = feature_values <= threshold
            right_mask = feature_values > threshold
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            gini_left = calculate_gini(y_left)
            gini_right = calculate_gini(y_right)
            
            weight_left = len(y_left) / n_samples
            weight_right = len(y_right) / n_samples
            
            gini_split = (weight_left * gini_left) + (weight_right * gini_right)
            info_gain = parent_gini - gini_split
            
            # Using strict inequality (> instead of >=) ensures that we break ties 
            # by keeping the smallest feature index and smallest threshold.
            if info_gain > best_gain:
                best_gain = info_gain
                best_feature = feature_idx
                best_threshold = threshold
                
    return [int(best_feature), float(best_threshold)]