def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    if not data or not data[0]:
        return data

    n_rows = len(data)
    n_cols = len(data[0])
    
    # Initialize a result matrix of the same dimensions with 0.0
    scaled_data = [[0.0 for _ in range(n_cols)] for _ in range(n_rows)]

    for j in range(n_cols):
        # Extract all values in the current column
        col_values = [data[i][j] for i in range(n_rows)]
        
        col_min = min(col_values)
        col_max = max(col_values)
        col_range = col_max - col_min

        for i in range(n_rows):
            if col_range == 0:
                # Requirement: If range is 0, set scaled values to 0.0
                scaled_data[i][j] = 0.0
            else:
                # Apply the Min-Max formula: (x - min) / (max - min)
                scaled_data[i][j] = float(data[i][j] - col_min) / col_range

    return scaled_data