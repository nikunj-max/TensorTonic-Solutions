def histogram_equalize(image):
    """
    Apply histogram equalization to enhance image contrast.
    """
    if not image or not image[0]:
        return image

    rows = len(image)
    cols = len(image[0])
    total_pixels = rows * cols

    # 1. Histogram
    hist = [0] * 256
    for r in range(rows):
        for c in range(cols):
            hist[image[r][c]] += 1

    # 2. CDF
    cdf = [0] * 256
    current_sum = 0
    for i in range(256):
        current_sum += hist[i]
        cdf[i] = current_sum

    # 3. CDF Min
    cdf_min = 0
    for val in cdf:
        if val > 0:
            cdf_min = val
            break

    # 4. Handle same-value edge case
    if total_pixels == cdf_min:
        return [[0 for _ in range(cols)] for _ in range(rows)]

    # 5. Mapping
    lut = [0] * 256
    denominator = total_pixels - cdf_min
    for i in range(256):
        numerator = cdf[i] - cdf_min
        lut[i] = int(round((numerator / denominator) * 255))

    # 6. Transform
    return [[lut[pixel] for pixel in row] for row in image]