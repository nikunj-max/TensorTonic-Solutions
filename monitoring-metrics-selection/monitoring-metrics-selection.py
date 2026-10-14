def compute_monitoring_metrics(system_type: str, y_true: list, y_pred: list) -> dict:
    """
    Returns a dictionary of metrics computed based on the system_type.
    """
    n = len(y_true)
    if n == 0:
        return {}

    if system_type == "classification":
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        
        accuracy = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
    elif system_type == "regression":
        mae = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n
        rmse = (sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n) ** 0.5
        
        return {
            "mae": mae,
            "rmse": rmse
        }
        
    elif system_type == "ranking":
        # Python's built-in sort is stable, ensuring tied predictions retain input order
        items = list(zip(y_true, y_pred))
        items.sort(key=lambda x: x[1], reverse=True)
        
        top_3 = items[:3]
        
        # Calculate true positives in top 3 (assuming relevant items have y_true == 1)
        relevant_in_top_3 = sum(yt for yt, yp in top_3)
        total_relevant = sum(y_true)
        
        precision_at_3 = relevant_in_top_3 / 3.0
        recall_at_3 = relevant_in_top_3 / total_relevant if total_relevant > 0 else 0.0
        
        return {
            "precision_at_3": precision_at_3,
            "recall_at_3": recall_at_3
        }

    else:
        raise ValueError(f"Unknown system_type: {system_type}")