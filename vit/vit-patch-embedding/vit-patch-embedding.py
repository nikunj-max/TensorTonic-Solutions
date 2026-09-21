import numpy as np

def patch_embed(image: np.ndarray, patch_size: int,
                W_proj: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Returns float64 patch embeddings with shape (B, N, D).
    """
    B, H, W, C = image.shape
    P = patch_size
    
    # 1. Crop image to whole multiples of patch_size (discarding bottom/right edges)
    H_crop = (H // P) * P
    W_crop = (W // P) * P
    cropped_image = image[:, :H_crop, :W_crop, :]
    
    # 2. Reshape to split the height and width into patch grids and internal patch dimensions
    H_p = H_crop // P
    W_p = W_crop // P
    grid = cropped_image.reshape((B, H_p, P, W_p, P, C))
    
    # 3. Transpose to group the patches properly: (B, H_p, W_p, P, P, C)
    # Axes: 0:Batch, 1:H_p, 3:W_p, 2:P(height), 4:P(width), 5:Channels
    patches = grid.transpose(0, 1, 3, 2, 4, 5)
    
    # 4. Flatten the 2D grid of patches into a 1D sequence (N) and flatten the patch pixels
    N = H_p * W_p
    patch_dim = P * P * C
    flattened_patches = patches.reshape((B, N, patch_dim))
    
    # 5. Apply linear projection and add bias
    embeddings = np.matmul(flattened_patches, W_proj) + bias
    
    # Ensure float64 dtype return
    return embeddings.astype(np.float64)