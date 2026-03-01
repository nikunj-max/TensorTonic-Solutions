import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    # Convert input to a numpy array
    x = np.asarray(x)
    n = len(x)
    
    # Initialize random generator
    if rng is None:
        rng = np.random.default_rng()
        
    # Generate bootstrap indices: (n_bootstrap, n)
    # We sample 'n' indices with replacement for each of the 'n_bootstrap' trials
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    
    # Create the bootstrap samples from the original data using the indices
    resamples = x[indices]
    
    # Calculate the mean for each bootstrap sample (along the rows)
    boot_means = np.mean(resamples, axis=1)
    
    # Calculate alpha and the corresponding quantiles
    alpha = 1 - ci
    lower_quantile = alpha / 2
    upper_quantile = 1 - (alpha / 2)
    
    # Estimate the confidence interval bounds
    lower = np.quantile(boot_means, lower_quantile)
    upper = np.quantile(boot_means, upper_quantile)
    
    return boot_means, lower, upper