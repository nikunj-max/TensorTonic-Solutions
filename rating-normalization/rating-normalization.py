def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    normalized_matrix = []
    
    for row in matrix:
        # Extract non-zero ratings to calculate the mean
        ratings = [val for val in row if val != 0]
        
        # If the user has no ratings, the row remains all 0.0
        if not ratings:
            normalized_matrix.append([0.0] * len(row))
            continue
            
        user_mean = sum(ratings) / len(ratings)
        
        # Subtract the mean from rated items, keep 0s as 0.0
        normalized_row = [
            float(val - user_mean) if val != 0 else 0.0 
            for val in row
        ]
        
        normalized_matrix.append(normalized_row)
        
    return normalized_matrix