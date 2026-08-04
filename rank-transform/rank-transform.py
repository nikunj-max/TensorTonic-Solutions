def rank_transform(values):
    """
    Replace each value with its average rank.
    """
    n = len(values)
    # Pair each value with its original index and sort by the value
    indexed_values = sorted((val, i) for i, val in enumerate(values))
    ranks = [0.0] * n
    
    i = 0
    while i < n:
        j = i
        # Find the end of the current group of tied values
        while j < n and indexed_values[j][0] == indexed_values[i][0]:
            j += 1
        
        # Calculate the average 1-based rank for the tied group
        # The group spans 1-based ranks from (i + 1) to j
        avg_rank = (i + 1 + j) / 2.0
        
        # Assign this average rank to the original indices of all tied elements
        for k in range(i, j):
            original_index = indexed_values[k][1]
            ranks[original_index] = float(avg_rank)
            
        # Move to the next unique value
        i = j
        
    return ranks