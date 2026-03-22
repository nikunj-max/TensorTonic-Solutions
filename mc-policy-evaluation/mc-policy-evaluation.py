import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    # Initialize sums and counters for each state
    v_sums = np.zeros(n_states)
    v_counts = np.zeros(n_states)
    
    for episode in episodes:
        g = 0  # Initialize return
        visited_states = set()
        
        # Prepare to store returns for the first-visit check
        # We calculate returns by iterating backwards: G_t = r_t + gamma * G_{t+1}
        returns_to_process = []
        
        for i in range(len(episode) - 1, -1, -1):
            state, reward = episode[i]
            g = reward + gamma * g
            returns_to_process.append((state, g))
            
        # Process returns in reverse (which is actually chronological order)
        # to identify the "first visit" easily with a set.
        # Alternatively, we can just check the set while going backwards
        # and keep updating the 'first' return seen for that state.
        
        first_visit_returns = {}
        for state, g_val in returns_to_process:
            # Since we are iterating from the end of the episode to the start,
            # the last time we see a state in this loop is actually its first visit.
            first_visit_returns[state] = g_val
            
        # Update global counts and sums
        for state, g_val in first_visit_returns.items():
            v_sums[state] += g_val
            v_counts[state] += 1
            
    # Calculate average, avoiding division by zero for unvisited states
    V = np.zeros(n_states)
    mask = v_counts > 0
    V[mask] = v_sums[mask] / v_counts[mask]
    
    return V
