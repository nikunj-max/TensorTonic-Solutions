def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Unpack coordinates
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    
    # Calculate area of both bounding boxes
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    
    # Find the coordinates of the intersection rectangle
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    
    # Calculate the width and height of the intersection
    # max(0, ...) ensures we don't get negative areas for non-overlapping boxes
    i_width = max(0.0, ix2 - ix1)
    i_height = max(0.0, iy2 - iy1)
    
    # Compute intersection area
    intersection = i_width * i_height
    
    # Compute union area
    union = area_a + area_b - intersection
    
    # Handle the edge case where both boxes have 0 area
    if union == 0:
        return 0.0
        
    # Return Intersection over Union
    return intersection / union