import itertools

def canonical_preference_pairs(records: list) -> dict:
    """
    Returns a dict with pairs: a list of prompt_id, winner_id, loser_id dictionaries.
    """
    pairs = []
    
    for record in records:
        prompt_id = record["prompt_id"]
        
        # Sort candidates by rank ascending (1 is best/winner, higher number is worse/loser)
        # Using sorted() creates a new list and leaves the original unmodified
        sorted_candidates = sorted(record["candidates"], key=lambda c: c["rank"])
        
        # itertools.combinations will yield pairs in the order of the sorted list,
        # which automatically satisfies ordering by winner rank then loser rank.
        for winner, loser in itertools.combinations(sorted_candidates, 2):
            pairs.append({
                "prompt_id": prompt_id,
                "winner_id": winner["id"],
                "loser_id": loser["id"]
            })
            
    return {"pairs": pairs}