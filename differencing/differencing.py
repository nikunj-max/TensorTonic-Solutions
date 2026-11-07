def differencing(series: list, order: int) -> list:
    """
    Returns the series after the requested differencing order.
    """
    # Create a copy of the series to avoid modifying the original input
    current_series = series[:]
    
    # Apply differencing 'order' times
    for _ in range(order):
        # Calculate the difference between consecutive elements
        current_series = [current_series[i+1] - current_series[i] for i in range(len(current_series) - 1)]
        
    return current_series