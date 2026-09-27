import numpy as np

def go_group_liberties(board: list, row: int, col: int) -> tuple:
    """
    Returns: a tuple of two sorted coordinate lists: group and liberties.
    """
    board_arr = np.array(board)
    rows, cols = board_arr.shape
    
    target_color = board_arr[row, col]
    
    group = set([(row, col)])
    liberties = set()
    stack = [(row, col)]
    
    # Orthogonal directions: Up, Down, Left, Right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while stack:
        r, c = stack.pop()
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # Check boundaries
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbor_val = board_arr[nr, nc]
                
                if neighbor_val == target_color:
                    if (nr, nc) not in group:
                        group.add((nr, nc))
                        stack.append((nr, nc))
                elif neighbor_val == 0:
                    liberties.add((nr, nc))
                    
    # Convert sets to sorted lists of lists (sorted automatically by row, then column by Python's sort)
    sorted_group = [list(coord) for coord in sorted(group)]
    sorted_liberties = [list(coord) for coord in sorted(liberties)]
    
    return sorted_group, sorted_liberties