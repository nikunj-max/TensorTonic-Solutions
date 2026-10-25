def local_vjp(operation: str, inputs: list, output: float, upstream_gradient: float) -> list:
    """
    Returns a list of float gradient contributions in input order.
    """
    # Ensure the upstream gradient is a float to guarantee float outputs
    g = float(upstream_gradient)
    
    if operation == "add":
        return [g, g]
    elif operation == "mul":
        a, b = float(inputs[0]), float(inputs[1])
        return [g * b, g * a]
    elif operation == "tanh":
        t = float(output)
        return [g * (1.0 - t ** 2)]
    else:
        raise ValueError(f"Unknown operation: {operation}")
