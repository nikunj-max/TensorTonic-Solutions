import numpy as np

def q_learning_update(Q: list, s: int, a: int, r: float, s_next: int, alpha: float, gamma: float) -> np.ndarray:
    """
    Applies a single tabular Q-learning update and returns the new Q-table.
    """
    # 1. Convert the input Q-table to a numpy array to avoid modifying the original list
    q_table = np.array(Q, dtype=float)
    
    # 2. Find the maximum Q-value for the next state (s_next) across all possible actions
    max_q_next = np.max(q_table[s_next])
    
    # 3. Calculate the TD target
    td_target = r + gamma * max_q_next
    
    # 4. Update the Q-value for the current state-action pair (s, a)
    q_table[s, a] = q_table[s, a] + alpha * (td_target - q_table[s, a])
    
    return q_table