def exponential_moving_average(values: list, alpha: float) -> list:
    """
    Returns the exponential moving average at every position.
    """
    ema = [values[0]]
    one_minus_alpha = 1 - alpha

    for x in values[1:]:
        ema.append(alpha * x + one_minus_alpha * ema[-1])

    return ema