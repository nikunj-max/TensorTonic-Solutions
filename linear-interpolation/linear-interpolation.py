def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    result = values[:]
    n = len(result)
    i = 0
    
    while i < n:
        if result[i] is None:
            left = i - 1
            right = i
            
            # Find the next non-None value
            while result[right] is None:
                right += 1
                
            v_left = result[left]
            v_right = result[right]
            
            # Interpolate for each missing value in the gap
            for j in range(left + 1, right):
                result[j] = v_left + (j - left) / (right - left) * (v_right - v_left)
                
            # Skip past the filled gap
            i = right
        else:
            i += 1
            
    return result