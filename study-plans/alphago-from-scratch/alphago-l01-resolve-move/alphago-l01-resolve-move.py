import numpy as np

def resolve_go_action(board: list, player: int, action: int, consecutive_passes: int, move_count: int) -> tuple:
    """
    Returns: a tuple: next board, player, captures, consecutive passes, move count, terminal.
    """
    board_arr = np.array(board)
    n = board_arr.shape[0]
    
    move_count += 1
    next_player = -player
    captures = 0
    
    if action == n * n:
        consecutive_passes += 1
    else:
        consecutive_passes = 0
        r, c = divmod(action, n)
        board_arr[r, c] = player
        
        # Helper to find group and its liberties
        def get_group_and_liberties(brd, start_r, start_c):
            color = brd[start_r, start_c]
            group = set([(start_r, start_c)])
            liberties = set()
            queue = [(start_r, start_c)]
            
            while queue:
                curr_r, curr_c = queue.pop(0)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        if brd[nr, nc] == 0:
                            liberties.add((nr, nc))
                        elif brd[nr, nc] == color and (nr, nc) not in group:
                            group.add((nr, nc))
                            queue.append((nr, nc))
            return group, liberties

        # Check adjacent intersections for opponent captures
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                # Need to check if it's STILL the opponent color (could have been captured in previous iteration)
                if board_arr[nr, nc] == -player:
                    group, liberties = get_group_and_liberties(board_arr, nr, nc)
                    if len(liberties) == 0:
                        captures += len(group)
                        for gr, gc in group:
                            board_arr[gr, gc] = 0
                            
    terminal = bool(consecutive_passes >= 2 or move_count >= 2 * n * n)
    
    return (board_arr, next_player, captures, consecutive_passes, move_count, terminal)