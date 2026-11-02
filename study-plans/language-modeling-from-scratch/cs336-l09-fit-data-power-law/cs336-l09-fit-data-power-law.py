import torch
import math

def fit_data_power_law(data_sizes: torch.Tensor, losses: torch.Tensor, target_sizes: torch.Tensor, floor_candidates: torch.Tensor) -> dict:
    """
    Fits a data scaling law L(D) = E + A * D^(-alpha) to observed data sizes and losses, 
    selecting the best irreducible-loss floor (E) from candidates.
    
    Returns a dict: floor, coefficient, exponent, predictions (float64 tensor), rmse.
    """
    best_rmse = float('inf')
    best_floor = None
    best_coeff = None
    best_exp = None
    
    # E must be strictly less than every observed loss
    min_loss = torch.min(losses).item()
    
    # Pre-compute log(D) and the design matrix X for the least squares solver
    x = torch.log(data_sizes)
    X = torch.stack([torch.ones_like(x), x], dim=1)
    
    for floor_t in floor_candidates:
        E = floor_t.item()
        
        if E >= min_loss:
            continue
            
        # log(L_i - E) = a + b * log(D_i)
        y = torch.log(losses - E)
        
        # Solve least-squares
        solution = torch.linalg.lstsq(X, y).solution
        a = solution[0].item()
        b = solution[1].item()
        
        try:
            A = math.exp(a)
        except OverflowError:
            continue
            
        alpha = -b
        
        # Coefficient and exponent must be strictly positive and finite
        if not (math.isfinite(A) and math.isfinite(alpha) and A > 0 and alpha > 0):
            continue
            
        # Calculate RMSE on the original loss scale
        preds = E + A * (data_sizes ** (-alpha))
        rmse = torch.sqrt(torch.mean(torch.square(preds - losses))).item()
        
        # Select best fit, strict less-than preserves order on exact ties
        if rmse < best_rmse:
            best_rmse = rmse
            best_floor = E
            best_coeff = A
            best_exp = alpha
            
    # Calculate predictions on the target sizes
    # Utilizing PyTorch tensor broadcasting matches target_sizes in shape and device
    predictions = best_floor + best_coeff * (target_sizes ** (-best_exp))
    
    return {
        "floor": float(best_floor),
        "coefficient": float(best_coeff),
        "exponent": float(best_exp),
        "rmse": float(best_rmse),
        "predictions": predictions
    }