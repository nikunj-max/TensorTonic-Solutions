import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns the normalized residual-block result and selected mode in a dictionary.
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    gamma1 = np.asarray(gamma1, dtype=np.float64)
    beta1 = np.asarray(beta1, dtype=np.float64)
    gamma2 = np.asarray(gamma2, dtype=np.float64)
    beta2 = np.asarray(beta2, dtype=np.float64)
    
    def batch_norm(val, gamma, beta):
        mean = np.mean(val, axis=0)
        var = np.var(val, axis=0)
        val_hat = (val - mean) / np.sqrt(var + 1e-5)
        return gamma * val_hat + beta

    def relu(val):
        return np.maximum(0, val)

    residual = x.copy()

    if mode == "post":
        out = x @ W1
        out = batch_norm(out, gamma1, beta1)
        out = relu(out)
        out = out @ W2
        out = batch_norm(out, gamma2, beta2)
        out = out + residual
        out = relu(out)
    elif mode == "pre":
        out = batch_norm(x, gamma1, beta1)
        out = relu(out)
        out = out @ W1
        out = batch_norm(out, gamma2, beta2)
        out = relu(out)
        out = out @ W2
        out = out + residual
    else:
        raise ValueError("Invalid mode. Expected 'post' or 'pre'.")

    # Round to four decimal places and convert to nested lists
    output_rounded = np.round(out, 4).tolist()
    
    return {
        "output": output_rounded,
        "mode": mode
    }