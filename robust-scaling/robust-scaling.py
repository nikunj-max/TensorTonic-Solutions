def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    n = len(values)
    if n == 1:
        return [0.0]

    sorted_vals = sorted(values)

    def get_median(arr):
        m = len(arr)
        if m % 2 != 0:
            return float(arr[m // 2])
        else:
            return (arr[m // 2 - 1] + arr[m // 2]) / 2.0

    median = get_median(sorted_vals)

    mid = n // 2
    if n % 2 != 0:
        lower_half = sorted_vals[:mid]
        upper_half = sorted_vals[mid + 1:]
    else:
        lower_half = sorted_vals[:mid]
        upper_half = sorted_vals[mid:]

    q1 = get_median(lower_half)
    q3 = get_median(upper_half)
    iqr = q3 - q1

    scaled_values = []
    for x in values:
        if iqr == 0:
            scaled_values.append(float(x - median))
        else:
            scaled_values.append(float((x - median) / iqr))

    return scaled_values