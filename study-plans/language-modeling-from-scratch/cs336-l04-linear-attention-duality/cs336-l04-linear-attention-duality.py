import torch


def linear_attention_duality(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict:
    orig_dtype = q.dtype
    B, S, Dk = q.shape
    Dv = v.shape[-1]

    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)

    # Parallel form
    scores = torch.tril(torch.bmm(q_f, k_f.transpose(1, 2)))
    parallel_output = torch.bmm(scores, v_f)

    # Recurrent form
    state = torch.zeros((B, Dk, Dv), dtype=torch.float32, device=q.device)
    outputs = []
    for t in range(S):
        k_t = k_f[:, t : t + 1, :].transpose(1, 2)
        v_t = v_f[:, t : t + 1, :]
        q_t = q_f[:, t : t + 1, :]
        state = state + torch.bmm(k_t, v_t)
        outputs.append(torch.bmm(q_t, state))

    recurrent_output = torch.cat(outputs, dim=1)

    return {
        "parallel_output": parallel_output.to(orig_dtype),
        "recurrent_output": recurrent_output.to(orig_dtype),
        "final_state": state.to(orig_dtype),
    }