def double_exponential_smoothing(series: list, alpha: float, beta: float) -> list:
    """
    Returns the smoothed level at every time step.
    """
    level = series[0]
    trend = series[1] - series[0]
    levels = [level]

    for y in series[1:]:
        prev_level = level
        level = alpha * y + (1 - alpha) * (prev_level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        levels.append(level)

    return levels