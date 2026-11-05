import torch

def muon_spectral_update(
    parameter: torch.Tensor, gradient: torch.Tensor,
    previous_momentum: torch.Tensor, momentum_coefficient: int | float,
    learning_rate: int | float,
) -> dict:
    """
    Returns a dict of tensors: new_parameter, new_momentum, orthogonalized_update.
    """
    # 1. Update the momentum (using the input's original dtype)
    new_momentum = momentum_coefficient * previous_momentum + gradient
    
    # 2. Cast momentum to float32 for SVD stability and precision
    momentum_f32 = new_momentum.to(torch.float32)
    
    # 3. Compute compact SVD (full_matrices=False ensures U is (m,r) and Vh is (r,n))
    U, S, Vh = torch.linalg.svd(momentum_f32, full_matrices=False)
    
    # 4. Orthogonalize by dropping the singular values (stretch) and multiplying U and V^T
    O_f32 = U @ Vh
    
    # 5. Restore the spectral update to the original dtype
    orthogonalized_update = O_f32.to(parameter.dtype)
    
    # 6. Apply the parameter update
    new_parameter = parameter - learning_rate * orthogonalized_update
    
    return {
        "new_parameter": new_parameter,
        "new_momentum": new_momentum,
        "orthogonalized_update": orthogonalized_update
    }