def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    num_states = len(values)
    new_values = [0.0] * num_states
    
    for s in range(num_states):
        action_values = []
        
        # Iterate through each action available for the current state
        for a in range(len(transitions[s])):
            # Calculate the immediate reward for this action
            q_sa = rewards[s][a]
            
            # Calculate the expected future value: sum(T(s, a, s') * V(s'))
            expected_future_value = 0.0
            for s_prime in range(len(transitions[s][a])):
                prob = transitions[s][a][s_prime]
                expected_future_value += prob * values[s_prime]
            
            # Add the discounted future value to the immediate reward
            q_sa += gamma * expected_future_value
            action_values.append(q_sa)
            
        # The new value for the state is the maximum of all possible action values
        new_values[s] = max(action_values)
        
    return new_values