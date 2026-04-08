def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Initialize a list of 256 zeros, one for each possible intensity (0-255)
    hist = [0] * 256
    
    # Iterate through each row in the 2D image
    for row in image:
        # Iterate through each pixel in the row
        for pixel in row:
            # Increment the bin corresponding to the pixel's intensity value
            hist[pixel] += 1
            
    return hist