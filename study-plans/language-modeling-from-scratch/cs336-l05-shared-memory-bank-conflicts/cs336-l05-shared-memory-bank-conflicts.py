import numpy as np

def bank_conflict_analysis(byte_addresses: np.ndarray, num_banks: int = 32, bank_width_bytes: int = 4) -> dict:
    """
    Returns a dict of int64 arrays: bank_ids and conflict_degree.
    """
    # Calculate bank ID for each lane
    bank_ids = (byte_addresses // bank_width_bytes) % num_banks
    bank_ids = bank_ids.astype(np.int64)
    
    # Initialize the conflict degree array
    conflict_degree = np.empty_like(bank_ids, dtype=np.int64)
    
    # Calculate conflict degree for each lane
    for i in range(len(byte_addresses)):
        b = bank_ids[i]
        # Find all addresses that fall into the same bank as the current lane
        same_bank_addrs = byte_addresses[bank_ids == b]
        # Count the number of distinct addresses in this bank
        conflict_degree[i] = len(np.unique(same_bank_addrs))
        
    return {
        "bank_ids": bank_ids,
        "conflict_degree": conflict_degree
    }