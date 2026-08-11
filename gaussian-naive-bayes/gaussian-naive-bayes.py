import math

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    n = len(y_train)
    num_features = len(X_train[0])
    
    classes = list(set(y_train))
    class_stats = {}
    
    # 1. Group data by class and compute prior, mean, and variance
    for c in classes:
        # Extract all training samples for class c
        X_c = [X_train[i] for i in range(n) if y_train[i] == c]
        n_c = len(X_c)
        
        log_prior = math.log(n_c / n)
        
        means = []
        variances = []
        
        for j in range(num_features):
            # Get feature j for all samples in class c
            col = [row[j] for row in X_c]
            
            # Compute mean
            mean_j = sum(col) / n_c
            
            # Compute population variance and add epsilon
            var_j = sum((val - mean_j) ** 2 for val in col) / n_c
            var_j += 1e-9 
            
            means.append(mean_j)
            variances.append(var_j)
            
        class_stats[c] = {
            'log_prior': log_prior,
            'means': means,
            'variances': variances
        }
        
    # 2. Predict the class for each test sample
    predictions = []
    
    for x in X_test:
        max_log_posterior = -float('inf')
        best_class = None
        
        for c in classes:
            stats = class_stats[c]
            log_posterior = stats['log_prior']
            
            # Add the log-likelihood of each feature
            for j in range(num_features):
                mean_j = stats['means'][j]
                var_j = stats['variances'][j]
                x_j = x[j]
                
                # Gaussian log-likelihood formula
                term1 = -0.5 * math.log(2 * math.pi * var_j)
                term2 = ((x_j - mean_j) ** 2) / (2 * var_j)
                log_posterior += (term1 - term2)
                
            if log_posterior > max_log_posterior:
                max_log_posterior = log_posterior
                best_class = c
                
        predictions.append(best_class)
        
    return predictions