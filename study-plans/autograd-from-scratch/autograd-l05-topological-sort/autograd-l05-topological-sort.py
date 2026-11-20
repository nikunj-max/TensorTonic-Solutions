import heapq


def topological_sort(nodes: list, output_id: str) -> list:
    """
    Returns a list of reachable node IDs in deterministic topological order.
    """
    pos = {}
    parents_map = {}
    for idx, node in enumerate(nodes):
        node_id = node["id"]
        pos[node_id] = idx
        parents_map[node_id] = node["parents"]

    # 1. Collect output_id and all of its ancestors via backward traversal
    reachable = set()
    stack = [output_id]
    while stack:
        curr = stack.pop()
        if curr not in reachable:
            reachable.add(curr)
            for parent in parents_map[curr]:
                if parent not in reachable:
                    stack.append(parent)

    # 2. Build forward adjacency list and in-degrees for the reachable subgraph
    children = {node_id: [] for node_id in reachable}
    in_degree = {}
    for node_id in reachable:
        parents = parents_map[node_id]
        in_degree[node_id] = len(parents)
        for parent in parents:
            children[parent].append(node_id)

    # 3. Kahn's algorithm with a min-heap keyed by original input position
    ready = [(pos[node_id], node_id) for node_id in reachable if in_degree[node_id] == 0]
    heapq.heapify(ready)

    order = []
    while ready:
        _, curr = heapq.heappop(ready)
        order.append(curr)
        for child in children[curr]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                heapq.heappush(ready, (pos[child], child))

    return order