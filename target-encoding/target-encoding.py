def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    sums = {}
    counts = {}
    
    # Accumulate sums and counts for each category
    for cat, target in zip(categories, targets):
        sums[cat] = sums.get(cat, 0) + target
        counts[cat] = counts.get(cat, 0) + 1
        
    # Calculate the mean for each category
    means = {cat: sums[cat] / counts[cat] for cat in sums}
    
    # Replace each original category with its computed mean
    return [means[cat] for cat in categories]