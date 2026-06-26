import math

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    pooled_outputs = []
    
    for roi in rois:
        x1, y1, x2, y2 = roi
        roi_h = y2 - y1
        roi_w = x2 - x1
        
        roi_grid = []
        for i in range(output_size):
            row = []
            for j in range(output_size):
                # Compute height boundaries
                hstart = y1 + math.floor((i * roi_h) / output_size)
                hend = y1 + math.floor(((i + 1) * roi_h) / output_size)
                if hend == hstart:
                    hend = hstart + 1
                
                # Compute width boundaries
                wstart = x1 + math.floor((j * roi_w) / output_size)
                wend = x1 + math.floor(((j + 1) * roi_w) / output_size)
                if wend == wstart:
                    wend = wstart + 1
                
                # Max pooling within the defined bin
                max_val = float('-inf')
                for r in range(hstart, hend):
                    for c in range(wstart, wend):
                        val = feature_map[r][c]
                        if val > max_val:
                            max_val = val
                
                row.append(max_val)
            roi_grid.append(row)
            
        pooled_outputs.append(roi_grid)
        
    return pooled_outputs