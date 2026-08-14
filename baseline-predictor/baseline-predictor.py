def baseline_predict(ratings_matrix, target_pairs):
    """
    Compute baseline predictions using global mean and user/item biases.
    """
    if not ratings_matrix or not ratings_matrix[0]:
        return []

    num_users = len(ratings_matrix)
    num_items = len(ratings_matrix[0])

    # 1. Compute global mean (mu)
    total_sum = 0
    total_count = 0
    for r in range(num_users):
        for c in range(num_items):
            if ratings_matrix[r][c] != 0:
                total_sum += ratings_matrix[r][c]
                total_count += 1
                
    mu = total_sum / total_count if total_count > 0 else 0

    # 2. Compute user biases
    user_biases = [0.0] * num_users
    for u in range(num_users):
        u_sum = 0
        u_count = 0
        for c in range(num_items):
            if ratings_matrix[u][c] != 0:
                u_sum += ratings_matrix[u][c]
                u_count += 1
        if u_count > 0:
            user_biases[u] = (u_sum / u_count) - mu

    # 3. Compute item biases
    item_biases = [0.0] * num_items
    for i in range(num_items):
        i_sum = 0
        i_count = 0
        for r in range(num_users):
            if ratings_matrix[r][i] != 0:
                i_sum += ratings_matrix[r][i]
                i_count += 1
        if i_count > 0:
            item_biases[i] = (i_sum / i_count) - mu

    # 4. Generate predictions for target pairs
    predictions = []
    for u, i in target_pairs:
        pred = mu + user_biases[u] + item_biases[i]
        predictions.append(pred)

    return predictions