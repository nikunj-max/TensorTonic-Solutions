def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.
    """
    H = len(image)
    W = len(image[0])
    
    resized_image = []
    
    for i in range(new_h):
        row = []
        for j in range(new_w):
            # Map output coordinates to source coordinates
            src_y = i * (H - 1) / (new_h - 1) if new_h > 1 else 0.0
            src_x = j * (W - 1) / (new_w - 1) if new_w > 1 else 0.0
            
            # Get integer (floor) and fractional parts
            y0 = int(src_y)
            x0 = int(src_x)
            dy = src_y - y0
            dx = src_x - x0
            
            # Clamp coordinates to stay within bounds
            y1 = min(y0 + 1, H - 1)
            x1 = min(x0 + 1, W - 1)
            
            # Interpolate from 4 nearest neighbors
            top_left = image[y0][x0] * (1 - dy) * (1 - dx)
            bottom_left = image[y1][x0] * dy * (1 - dx)
            top_right = image[y0][x1] * (1 - dy) * dx
            bottom_right = image[y1][x1] * dy * dx
            
            # Sum up to get the final pixel value
            pixel_val = top_left + bottom_left + top_right + bottom_right
            row.append(float(pixel_val))
            
        resized_image.append(row)
        
    return resized_image