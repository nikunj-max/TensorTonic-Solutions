def lag_features(series, lags):
    """
    Create a lag feature matrix from the time series.
    """
    max_lag = max(lags)
    feature_matrix = []
    
    for t in range(max_lag, len(series)):
        row = [series[t - lag] for lag in lags]
        feature_matrix.append(row)
        
    return feature_matrix