def nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression.
    """
    if not boxes:
        return []

    # Sort indices based on confidence scores in descending order
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept_indices = []

    def compute_iou(boxA, boxB):
        x1A, y1A, x2A, y2A = boxA
        x1B, y1B, x2B, y2B = boxB

        # Determine the coordinates of the intersection rectangle
        x1_inter = max(x1A, x1B)
        y1_inter = max(y1A, y1B)
        x2_inter = min(x2A, x2B)
        y2_inter = min(y2A, y2B)

        # Calculate intersection area
        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height

        if inter_area == 0:
            return 0.0

        # Calculate areas of both bounding boxes
        areaA = (x2A - x1A) * (y2A - y1A)
        areaB = (x2B - x1B) * (y2B - y1B)

        # Compute IoU
        union_area = areaA + areaB - inter_area
        if union_area == 0:
            return 0.0
            
        return inter_area / union_area

    while indices:
        # Pick the box with the highest score
        current_idx = indices.pop(0)
        kept_indices.append(current_idx)

        # Keep only the boxes that have IoU less than the threshold
        remaining_indices = []
        for idx in indices:
            iou = compute_iou(boxes[current_idx], boxes[idx])
            if iou < iou_threshold:
                remaining_indices.append(idx)
        
        indices = remaining_indices

    return kept_indices