def interaction_features(X):
    """
    Generate pairwise interaction features and append them to the original features.
    """
    result = []
    for row in X:
        d = len(row)
        # Start with the original features
        new_row = list(row)
        
        # Calculate pairwise interactions
        for i in range(d):
            for j in range(i + 1, d):
                new_row.append(row[i] * row[j])
                
        result.append(new_row)
        
    return result