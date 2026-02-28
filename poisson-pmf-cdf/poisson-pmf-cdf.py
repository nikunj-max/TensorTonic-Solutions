import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF using stable log-factorial computation.
    """
    def get_log_factorial(n):
        if n <= 1:
            return 0.0
        return np.sum(np.log(np.arange(1, n + 1)))

    # Compute PMF for the specific k
    # Using the log-space formula: exp(-lam + k * log(lam) - log(k!))
    log_pmf_k = -lam + k * np.log(lam) - get_log_factorial(k)
    pmf = np.exp(log_pmf_k)

    # Compute CDF by summing PMF values from 0 to k
    # We pre-calculate all log factorials up to k to be efficient
    cdf = 0.0
    # Pre-calculating log factorials using cumsum for efficiency
    if k == 0:
        cdf = pmf
    else:
        log_factorials = np.zeros(k + 1)
        log_factorials[1:] = np.cumsum(np.log(np.arange(1, k + 1)))
        
        # Calculate PMFs for all i in [0, k]
        # P(X=i) = exp(-lam + i*log(lam) - log_factorials[i])
        i_values = np.arange(k + 1)
        all_log_pmfs = -lam + i_values * np.log(lam) - log_factorials
        cdf = np.sum(np.exp(all_log_pmfs))

    return float(pmf), float(cdf)