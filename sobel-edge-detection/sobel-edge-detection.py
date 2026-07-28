import math

def sobel_edges(image):
    """
    Apply the Sobel operator to detect edges.
    """
    if not image or not image[0]:
        return []

    rows = len(image)
    cols = len(image[0])
    
    # Create zero-padded image
    padded = [[0] * (cols + 2) for _ in range(rows + 2)]
    for r in range(rows):
        for c in range(cols):
            padded[r + 1][c + 1] = image[r][c]
            
    # Sobel kernels
    Kx = [[-1, 0, 1], 
          [-2, 0, 2], 
          [-1, 0, 1]]
          
    Ky = [[-1, -2, -1], 
          [ 0,  0,  0], 
          [ 1,  2,  1]]
          
    # Initialize the output matrix with zeros
    result = [[0.0] * cols for _ in range(rows)]
    
    # Apply the convolution over each pixel
    for r in range(rows):
        for c in range(cols):
            gx = 0.0
            gy = 0.0
            
            # 3x3 window over the padded image
            for i in range(3):
                for j in range(3):
                    pixel = padded[r + i][c + j]
                    gx += pixel * Kx[i][j]
                    gy += pixel * Ky[i][j]
                    
            # Compute the gradient magnitude
            result[r][c] = math.sqrt(gx**2 + gy**2)
            
    return result