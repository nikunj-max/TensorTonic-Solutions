import torch


def mamba2_gated_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: torch.Tensor,
) -> dict:
    """
    Returns a dict of tensors: outputs and final_state.
    """
    out_dtype = q.dtype
    device = q.device
    B, S, Dk = q.shape
    Dv = v.shape[-1]

    # Upcast to float32 for numerical stability without mutating inputs
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    gamma_f = gamma.to(torch.float32)

    state = torch.zeros((B, Dk, Dv), dtype=torch.float32, device=device)
    outputs = []

    for t in range(S):
        g_t = gamma_f[:, t].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        k_t = k_f[:, t, :].unsqueeze(-1)                 # (B, Dk, 1)
        v_t = v_f[:, t, :].unsqueeze(1)                  # (B, 1, Dv)
        q_t = q_f[:, t, :].unsqueeze(1)                  # (B, 1, Dk)

        # S_t = gamma_t * S_{t-1} + k_t @ v_t^T
        state = g_t * state + torch.bmm(k_t, v_t)        # (B, Dk, Dv)

        # y_t = q_t^T @ S_t
        y_t = torch.bmm(q_t, state).squeeze(1)           # (B, Dv)
        outputs.append(y_t)

    return {
        "outputs": torch.stack(outputs, dim=1).to(out_dtype),
        "final_state": state.to(out_dtype),
    }