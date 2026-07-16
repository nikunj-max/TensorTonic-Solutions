import math

def winsorize(values, lower_pct, upper_pct):
    """
    Clip values at the given percentile bounds.
    """
    n = len(values)
    if n == 0:
        return []
        
    sorted_vals = sorted(values)
    
    def compute_percentile(p):
        k = (n - 1) * (p / 100.0)
        floor_k = math.floor(k)
        ceil_k = math.ceil(k)
        
        if floor_k == ceil_k:
            return float(sorted_vals[int(k)])
        
        p_val = sorted_vals[floor_k] + (k - floor_k) * (sorted_vals[ceil_k] - sorted_vals[floor_k])
        return float(p_val)
        
    lower_bound = compute_percentile(lower_pct)
    upper_bound = compute_percentile(upper_pct)
    
    return [float(max(lower_bound, min(upper_bound, v))) for v in values]