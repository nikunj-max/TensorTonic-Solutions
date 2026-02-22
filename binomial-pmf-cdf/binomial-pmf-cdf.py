import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # PMF: P(X = k) = comb(n, k) * (p**k) * ((1-p)**(n-k))
    pmf = float(comb(n, k) * (p**k) * ((1 - p)**(n - k)))
    
    # CDF: P(X <= k) = sum of PMF from 0 to k
    cdf = 0.0
    for i in range(k + 1):
        cdf += comb(n, i) * (p**i) * ((1 - p)**(n - i))
        
    return pmf, float(cdf)