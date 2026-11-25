def discount_returns(rewards: list, gamma: float) -> list:
    """
    Returns the discounted return at every timestep.
    """
    returns = []
    append = returns.append
    running = 0.0

    for r in reversed(rewards):
        running = r + gamma * running
        append(round(running, 4))

    returns.reverse()
    return returns