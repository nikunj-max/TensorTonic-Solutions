import math

def adjusted_cosine_similarity(ratings_matrix: list, item_i: int, item_j: int) -> float:
    """
    Returns the adjusted cosine similarity between the requested items.
    """
    numerator = 0.0
    sum_sq_i = 0.0
    sum_sq_j = 0.0
    
    for user_ratings in ratings_matrix:
        r_ui = user_ratings[item_i]
        r_uj = user_ratings[item_j]
        
        # Include user only if they rated both requested items (nonzero)
        if r_ui != 0 and r_uj != 0:
            # Calculate the user's mean from ALL their nonzero ratings
            nonzero_ratings = [r for r in user_ratings if r != 0]
            r_bar_u = sum(nonzero_ratings) / len(nonzero_ratings)
            
            # Center the ratings around the user's mean
            c_i = r_ui - r_bar_u
            c_j = r_uj - r_bar_u
            
            # Accumulate the cross-product and square sums
            numerator += c_i * c_j
            sum_sq_i += c_i ** 2
            sum_sq_j += c_j ** 2
            
    # Calculate the denominator by taking the square root of each square sum separately
    denominator = math.sqrt(sum_sq_i) * math.sqrt(sum_sq_j)
    
    # Return 0.0 if the denominator is zero to avoid division by zero
    if denominator == 0.0:
        return 0.0
        
    return float(numerator / denominator)