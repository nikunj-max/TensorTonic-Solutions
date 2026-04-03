def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    Formula: Y = 0.299*R + 0.587*G + 0.114*B
    """
    # Get dimensions: Height (H) and Width (W)
    height = len(image)
    width = len(image[0])
    
    # Initialize the 2D grayscale matrix
    grayscale_image = []
    
    for i in range(height):
        row = []
        for j in range(width):
            # Extract R, G, B components
            r = image[i][j][0]
            g = image[i][j][1]
            b = image[i][j][2]
            
            # Apply the luminance-weighted sum
            # Using floats for precision as required
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            row.append(luminance)
            
        grayscale_image.append(row)
        
    return grayscale_image