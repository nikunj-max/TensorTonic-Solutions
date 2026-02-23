import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Convert input to a numpy array to ensure vectorized operations
    C = np.array(C, dtype=float)
    
    # Calculate row totals and column totals
    row_sums = np.sum(C, axis=1)
    col_sums = np.sum(C, axis=0)
    total_sum = np.sum(C)
    
    # Compute expected frequencies: (row_total * col_total) / grand_total
    # np.outer handles the multiplication of every row sum with every col sum
    expected = np.outer(row_sums, col_sums) / total_sum
    
    # Calculate Chi-Square statistic: sum((O - E)^2 / E)
    chi2 = np.sum((C - expected) ** 2 / expected)
    
    return chi2, expected