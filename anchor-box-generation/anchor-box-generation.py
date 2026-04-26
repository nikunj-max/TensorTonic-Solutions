import math

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    
    Args:
        feature_size (int): Size of the square feature grid (e.g., 3 for a 3x3 grid).
        image_size (int): Size of the original square image in pixels.
        scales (list of float): List of anchor scales.
        aspect_ratios (list of float): List of aspect ratios (width/height).
        
    Returns:
        list of list of float: List of [x1, y1, x2, y2] anchor boxes.
    """
    anchors = []
    
    # 1. Compute the stride (spacing between grid cells in image space)
    stride = image_size / feature_size
    
    # 2. Iterate over grid cells in row-major order (i then j)
    for i in range(feature_size):
        for j in range(feature_size):
            # Compute the center of the current cell in image coordinates
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            
            # 3. For each cell, iterate over scales then aspect ratios
            for s in scales:
                for r in aspect_ratios:
                    # Compute box width and height based on scale and aspect ratio
                    # w = s * sqrt(r), h = s / sqrt(r)
                    sqrt_r = math.sqrt(r)
                    w = s * sqrt_r
                    h = s / sqrt_r
                    
                    # 4. Define the anchor box as [x1, y1, x2, y2]
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    anchors.append([x1, y1, x2, y2])
                    
    return anchors