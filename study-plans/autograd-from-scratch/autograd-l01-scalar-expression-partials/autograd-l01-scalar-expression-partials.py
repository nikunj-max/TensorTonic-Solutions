import numpy as np

def scalar_expression_partials(a: float, b: float, c: float, h: float) -> tuple[float, float, float, float]:
    """
    Returns the expression value and numerical partials for a, b, and c
    using forward-differences via list perturbations.
    """
    # Use explicit 64-bit floats to ensure precision matches expectations
    h_np = np.float64(h)
    inputs = [np.float64(a), np.float64(b), np.float64(c)]
    
    # Define the scalar expression operating on the list
    def d(vars_list):
        return vars_list[0] * vars_list[1] + vars_list[2]
        
    base_val = d(inputs)
    partials = []
    
    # Hint 1 & 2: Use enumerate and list.copy to isolate perturbations
    for i, val in enumerate(inputs):
        perturbed = inputs.copy()
        perturbed[i] += h_np
        
        # Calculate the forward-difference for this specific variable
        derivative = (d(perturbed) - base_val) / h_np
        partials.append(derivative)
        
    return (float(base_val), float(partials[0]), float(partials[1]), float(partials[2]))