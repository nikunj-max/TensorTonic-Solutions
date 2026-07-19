def mean_rating_imputation(ratings_matrix, mode):
    """
    Fill missing ratings (zeros) with user or item means.
    """
    if not ratings_matrix or not ratings_matrix[0]:
        return []

    rows = len(ratings_matrix)
    cols = len(ratings_matrix[0])
    
    # Create a new matrix to avoid modifying the input
    imputed = [[val for val in row] for row in ratings_matrix]

    if mode == "user":
        for i in range(rows):
            # Extract non-zero ratings for the user
            non_zeros = [val for val in ratings_matrix[i] if val != 0]
            if non_zeros:
                user_mean = sum(non_zeros) / len(non_zeros)
                # Fill missing values
                for j in range(cols):
                    if imputed[i][j] == 0:
                        imputed[i][j] = float(user_mean)
                        
    elif mode == "item":
        for j in range(cols):
            # Extract non-zero ratings for the item across all users
            non_zeros = [ratings_matrix[i][j] for i in range(rows) if ratings_matrix[i][j] != 0]
            if non_zeros:
                item_mean = sum(non_zeros) / len(non_zeros)
                # Fill missing values
                for i in range(rows):
                    if imputed[i][j] == 0:
                        imputed[i][j] = float(item_mean)

    return imputed