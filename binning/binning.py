def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    if not values:
        return []
        
    min_val = min(values)
    max_val = max(values)
    
    if min_val == max_val:
        return [0] * len(values)
        
    width = (max_val - min_val) / num_bins
    result = []
    
    for v in values:
        bin_idx = int((v - min_val) / width)
        # Clamp the maximum value to the last bin
        bin_idx = min(bin_idx, num_bins - 1)
        result.append(bin_idx)
        
    return result