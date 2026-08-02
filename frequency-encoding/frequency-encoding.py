def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    # Write code here
    total = len(values)
    counts = {}
    
    # Count occurrences of each unique value
    for val in values:
        counts[val] = counts.get(val, 0) + 1
        
    # Replace each value with its proportion and return
    return [counts[val] / total for val in values]