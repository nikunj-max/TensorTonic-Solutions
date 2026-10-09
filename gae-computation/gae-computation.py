def gae(rewards: list, values: list, gamma: float, lam: float) -> list:
    """
    Returns the generalized advantage estimate at every timestep.
    """
    T = len(rewards)
    advantages = [0.0] * T
    
    next_advantage = 0.0
    
    # Traverse timesteps backward
    for t in range(T - 1, -1, -1):
        # Compute the TD error
        delta_t = rewards[t] + gamma * values[t+1] - values[t]
        
        # Compute the advantage using the recursive formula
        advantage_t = delta_t + gamma * lam * next_advantage
        
        # Store current advantage and carry it to the next backward step
        advantages[t] = advantage_t
        next_advantage = advantage_t
        
    return advantages