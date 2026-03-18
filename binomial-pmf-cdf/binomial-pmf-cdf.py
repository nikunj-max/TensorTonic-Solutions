import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Calculate Probability Mass Function (PMF) for X = k
    # P(X = k) = comb(n, k) * (p**k) * ((1-p)**(n-k))
    pmf = float(comb(n, k) * (p**k) * ((1 - p)**(n - k)))
    
    # Calculate Cumulative Distribution Function (CDF) for X <= k
    # We sum the PMF values for all i from 0 to k
    cdf = 0.0
    for i in range(k + 1):
        cdf += comb(n, i) * (p**i) * ((1 - p)**(n - i))
        
    return pmf, float(cdf)