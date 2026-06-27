def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    n = len(y_true)
    if n == 0:
        return 0.0
        
    bin_counts = [0] * n_bins
    bin_true_sums = [0.0] * n_bins
    bin_pred_sums = [0.0] * n_bins
    
    # Assign each sample to a bin and accumulate statistics
    for y, p in zip(y_true, y_pred):
        # Find bin index; handle the p == 1.0 edge case
        bin_idx = min(int(p * n_bins), n_bins - 1)
        
        bin_counts[bin_idx] += 1
        bin_true_sums[bin_idx] += y
        bin_pred_sums[bin_idx] += p
        
    ece = 0.0
    
    # Compute the weighted difference for each non-empty bin
    for i in range(n_bins):
        if bin_counts[i] > 0:
            acc = bin_true_sums[i] / bin_counts[i]
            conf = bin_pred_sums[i] / bin_counts[i]
            weight = bin_counts[i] / n
            
            ece += weight * abs(acc - conf)
            
    return ece