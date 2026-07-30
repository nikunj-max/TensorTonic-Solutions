import math

def rotate_image(image, angle_degrees):
    """
    Rotate the image counterclockwise by the given angle using nearest neighbor interpolation.
    """
    if not image or not image[0]:
        return []
    
    H = len(image)
    W = len(image[0])
    
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    
    theta = math.radians(angle_degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    output = [[0 for _ in range(W)] for _ in range(H)]
    
    for i in range(H):
        for j in range(W):
            dy = i - cy
            dx = j - cx
            
            src_y = cy + dy * cos_t + dx * sin_t
            src_x = cx - dy * sin_t + dx * cos_t
            
            sy = round(src_y)
            sx = round(src_x)
            
            if 0 <= sy < H and 0 <= sx < W:
                output[i][j] = image[sy][sx]
                
    return output