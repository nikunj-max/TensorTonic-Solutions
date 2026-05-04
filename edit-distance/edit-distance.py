def edit_distance(s1, s2):
    """
    Compute the minimum edit distance between two strings.
    """
    m, n = len(s1), len(s2)
    
    # Create a DP table of size (m+1) x (n+1)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Initialize base cases: transforming to/from an empty string
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # If characters match, no operation is needed
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Take the minimum of delete, insert, or replace operations
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # Delete
                    dp[i][j - 1],    # Insert
                    dp[i - 1][j - 1] # Replace
                )
                
    return dp[m][n]