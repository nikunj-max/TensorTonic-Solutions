def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    if not models:
        return None
        
    # Sort criteria tuple for max():
    # 1. x["accuracy"]  -> Higher is better (default max behavior)
    # 2. -x["latency"]  -> Lower is better (negated so smaller latency becomes a "larger" negative number)
    # 3. x["timestamp"] -> Latest is better (ISO strings compare chronologically, later date = "larger" string)
    best_model = max(
        models,
        key=lambda x: (x["accuracy"], -x["latency"], x["timestamp"])
    )
    
    return best_model["name"]