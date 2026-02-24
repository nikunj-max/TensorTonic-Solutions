import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # Convert input list to a NumPy array
    x = np.array(x)
    
    # Calculate sample mean (x̄)
    sample_mean = np.mean(x)
    
    # Calculate sample standard deviation (s) with Bessel's correction (ddof=1)
    # This uses n-1 in the denominator
    sample_std = np.std(x, ddof=1)
    
    # Calculate sample size (n)
    n = len(x)
    
    # Calculate the Standard Error (SE = s / sqrt(n))
    standard_error = sample_std / np.sqrt(n)
    
    # Compute the t-statistic: t = (x̄ - μ₀) / SE
    t_stat = (sample_mean - mu0) / standard_error
    
    return float(t_stat)