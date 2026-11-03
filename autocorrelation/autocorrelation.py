def autocorrelation(series: list, max_lag: int) -> list:
    """
    Returns normalized autocorrelation from lag zero through max_lag.
    """
    n = len(series)
    mean = sum(series) / n
    centered = [x - mean for x in series]
    gamma_0 = sum(c * c for c in centered)

    # Handle constant series (zero variance)
    if gamma_0 == 0:
        return [1.0] + [0.0] * max_lag

    result = []
    for k in range(max_lag + 1):
        cov_k = sum(centered[t] * centered[t + k] for t in range(n - k))
        val = round(cov_k / gamma_0, 6)
        # Normalize -0.0 to 0.0 if produced by rounding
        result.append(0.0 if val == 0.0 else val)

    return result