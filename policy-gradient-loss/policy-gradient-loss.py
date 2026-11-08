def policy_gradient_loss(log_probs: list, rewards: list, gamma: float) -> float:
    """
    Returns the REINFORCE loss with a mean-return baseline.
    """
    T = len(rewards)
    returns = [0.0] * T
    
    # 1. Compute discounted returns backward
    returns[-1] = float(rewards[-1])
    for t in range(T - 2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]
        
    # 2. Subtract the mean return as a baseline to reduce variance
    mean_return = sum(returns) / T
    advantages = [G - mean_return for G in returns]
    
    # 3. Compute the loss (negative because we want gradient ascent on expected return)
    loss = -sum(lp * adv for lp, adv in zip(log_probs, advantages)) / T
    
    return float(loss)