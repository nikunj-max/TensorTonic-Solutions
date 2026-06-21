import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_test, d = X_test.shape
    n_train = len(y_train)
    
    log_priors = np.zeros(n_classes)
    # theta[c, i] will store P(x_i = 1 | class c)
    theta = np.zeros((n_classes, d))
    
    for idx, c in enumerate(classes):
        # Subset training samples belonging to class c
        X_c = X_train[y_train == c]
        n_c = X_c.shape[0]
        
        # Class prior with MLE (unsmoothed per standard definition/requirements)
        log_priors[idx] = np.log(n_c / n_train)
        
        # Feature counts for class c where x_i == 1
        feature_counts = np.sum(X_c, axis=0)
        
        # Laplace smoothing (alpha = 1)
        theta[idx] = (feature_counts + 1) / (n_c + 2)
        
    # To compute log likelihood efficiently for all test samples:
    # log P(x_i | y) = x_i * log(theta) + (1 - x_i) * log(1 - theta)
    log_theta = np.log(theta)
    log_one_minus_theta = np.log(1.0 - theta)
    
    # Initialize output array
    log_posteriors = np.zeros((n_test, n_classes))
    
    for idx in range(n_classes):
        # Broadcast across test instances
        # X_test shape: (n_test, d), log_theta[idx] shape: (d,)
        log_likelihood = X_test @ log_theta[idx] + (1 - X_test) @ log_one_minus_theta[idx]
        log_posteriors[:, idx] = log_priors[idx] + log_likelihood
        
    return log_posteriors