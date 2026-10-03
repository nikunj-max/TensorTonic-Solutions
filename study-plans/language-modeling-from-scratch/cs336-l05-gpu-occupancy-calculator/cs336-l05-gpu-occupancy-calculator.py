def gpu_occupancy(
    threads_per_block: int, registers_per_thread: int, shared_mem_per_block: int,
    max_threads_per_sm: int, max_warps_per_sm: int, max_blocks_per_sm: int,
    max_registers_per_sm: int, max_shared_mem_per_sm: int, warp_size: int = 32,
) -> dict:
    """
    Returns a dict: blocks_per_sm (int), resident_warps (int), occupancy (float).
    """
    # Calculate allocated warps per block and the effective thread capacity consumed
    warps_per_block = (threads_per_block + warp_size - 1) // warp_size
    effective_threads_per_block = warps_per_block * warp_size
    
    # Calculate block limits for each hardware constraint
    blocks_limit_threads = max_threads_per_sm // effective_threads_per_block
    blocks_limit_warps = max_warps_per_sm // warps_per_block
    blocks_limit_blocks = max_blocks_per_sm
    
    if registers_per_thread > 0:
        regs_per_block = registers_per_thread * effective_threads_per_block
        blocks_limit_regs = max_registers_per_sm // regs_per_block
    else:
        blocks_limit_regs = float('inf')
        
    if shared_mem_per_block > 0:
        blocks_limit_smem = max_shared_mem_per_sm // shared_mem_per_block
    else:
        blocks_limit_smem = float('inf')
        
    # Maximum resident blocks is the bottleneck (minimum) of all constraints
    blocks_per_sm = min(
        blocks_limit_threads,
        blocks_limit_warps,
        blocks_limit_blocks,
        blocks_limit_regs,
        blocks_limit_smem
    )
    
    # Ensure zero if any limit fails immediately
    blocks_per_sm = max(0, int(blocks_per_sm))
    
    resident_warps = blocks_per_sm * warps_per_block
    occupancy = resident_warps / max_warps_per_sm
    
    return {
        "blocks_per_sm": blocks_per_sm,
        "resident_warps": resident_warps,
        "occupancy": float(occupancy)
    }