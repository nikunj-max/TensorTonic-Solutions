def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    # Get dimensions of the input image and kernel
    h_in = len(image)
    w_in = len(image[0])
    kh = len(kernel)
    kw = len(kernel[0])

    # 1. Apply Zero Padding
    # Create a new grid of zeros with padded dimensions
    h_padded = h_in + 2 * padding
    w_padded = w_in + 2 * padding
    padded_img = [[0.0 for _ in range(w_padded)] for _ in range(h_padded)]
    
    # Copy the original image into the center of the padded grid
    for r in range(h_in):
        for c in range(w_in):
            padded_img[r + padding][c + padding] = float(image[r][c])

    # 2. Calculate Output Dimensions
    # Using the formula: floor((Input + 2*Padding - Kernel) / Stride) + 1
    h_out = ((h_in + 2 * padding - kh) // stride) + 1
    w_out = ((w_in + 2 * padding - kw) // stride) + 1
    
    # Initialize the output matrix
    output = [[0.0 for _ in range(w_out)] for _ in range(h_out)]

    # 3. Perform Convolution
    for i in range(h_out):
        for j in range(w_out):
            # Determine the top-left corner of the current receptive field in the padded image
            curr_r = i * stride
            curr_c = j * stride
            
            # Weighted sum (dot product) of the kernel and the image patch
            val = 0.0
            for m in range(kh):
                for n in range(kw):
                    val += padded_img[curr_r + m][curr_c + n] * kernel[m][n]
            
            output[i][j] = val

    return output