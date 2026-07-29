def morphological_op(image, kernel, operation):
    """
    Apply morphological erosion or dilation to a binary image.
    """
    if not image or not image[0]:
        return []

    img_h = len(image)
    img_w = len(image[0])
    k_h = len(kernel)
    k_w = len(kernel[0])

    pad_h = k_h // 2
    pad_w = k_w // 2

    # Step 1: Create a zero-padded image
    padded = [[0] * (img_w + 2 * pad_w) for _ in range(img_h + 2 * pad_h)]
    for i in range(img_h):
        for j in range(img_w):
            padded[i + pad_h][j + pad_w] = image[i][j]

    # Step 2: Extract active kernel positions relative to the kernel center
    # This optimizes the nested loops by only checking where kernel == 1
    active_offsets = []
    for i in range(k_h):
        for j in range(k_w):
            if kernel[i][j] == 1:
                # Store relative offsets from the center (pad_h, pad_w)
                active_offsets.append((i - pad_h, j - pad_w))

    # Step 3: Apply the morphological operation
    output = [[0] * img_w for _ in range(img_h)]

    for i in range(img_h):
        for j in range(img_w):
            center_i = i + pad_h
            center_j = j + pad_w

            if operation == "erode":
                # Erosion: All active kernel positions must match 1 in the padded image
                is_match = 1
                for di, dj in active_offsets:
                    if padded[center_i + di][center_j + dj] == 0:
                        is_match = 0
                        break  # Early exit on first failure
                output[i][j] = is_match

            elif operation == "dilate":
                # Dilation: Any active kernel position matching 1 is sufficient
                is_match = 0
                for di, dj in active_offsets:
                    if padded[center_i + di][center_j + dj] == 1:
                        is_match = 1
                        break  # Early exit on first match
                output[i][j] = is_match

    return output