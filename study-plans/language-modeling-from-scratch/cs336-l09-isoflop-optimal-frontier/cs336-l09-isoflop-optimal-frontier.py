import torch
import math

def isoflop_optimal_frontier(compute_budgets: torch.Tensor, parameter_counts: torch.Tensor, terminal_losses: torch.Tensor) -> dict:
    optima = []
    
    log_N_opt_list = []
    log_C_list = []
    
    for i in range(len(compute_budgets)):
        C = float(compute_budgets[i].item())
        
        # Ensure float64 arithmetic for precision
        N = parameter_counts[i].to(torch.float64)
        L = terminal_losses[i].to(torch.float64)
        
        # Independent variable is log(N)
        x = torch.log(N)
        
        # Design matrix for quadratic: x^2, x, 1
        X = torch.stack([x**2, x, torch.ones_like(x)], dim=1)
        
        # Solve least-squares to fit the quadratic L_C(x) = a_C*x^2 + b_C*x + c_C
        coeff = torch.linalg.lstsq(X, L).solution
        
        a = float(coeff[0].item())
        b = float(coeff[1].item())
        c = float(coeff[2].item())
        
        # Minimum of the quadratic
        x_opt = -b / (2.0 * a)
        N_opt = math.exp(x_opt)
        D_opt = C / (6.0 * N_opt)
        
        loss_opt = a * x_opt**2 + b * x_opt + c
        
        optima.append({
            "compute": C,
            "parameters": float(N_opt),
            "tokens": float(D_opt),
            "loss": float(loss_opt)
        })
        
        log_N_opt_list.append(x_opt)
        log_C_list.append(math.log(C))
        
    # Fit the frontier log(N_C) = log(k) + alpha * log(C)
    log_C_tensor = torch.tensor(log_C_list, dtype=torch.float64)
    log_N_tensor = torch.tensor(log_N_opt_list, dtype=torch.float64)
    
    # Design matrix for linear fit: log(C), 1
    X_line = torch.stack([log_C_tensor, torch.ones_like(log_C_tensor)], dim=1)
    coeff_line = torch.linalg.lstsq(X_line, log_N_tensor).solution
    
    alpha = float(coeff_line[0].item())
    log_k = float(coeff_line[1].item())
    k = math.exp(log_k)
    
    return {
        "optima": optima,
        "frontier_scale": float(k),
        "frontier_exponent": float(alpha)
    }