import math

def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian blur kernel.
    """
    kernel = [[0.0] * size for _ in range(size)]
    center = size // 2
    total_sum = 0.0

    # 1. Compute unnormalized weights
    for i in range(size):
        for j in range(size):
            # Calculate offsets from the center
            x = j - center
            y = i - center
            
            # Compute Gaussian function: G(x, y) = e^(- (x^2 + y^2) / (2 * sigma^2))
            exponent = -(x**2 + y**2) / (2 * (sigma**2))
            weight = math.exp(exponent)
            
            kernel[i][j] = weight
            total_sum += weight

    # 2. Normalize the kernel so all entries sum to 1.0
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= total_sum

    return kernel