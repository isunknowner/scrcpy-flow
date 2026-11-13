from __future__ import annotations


def compute_layout(nodes, edges) -> dict[int, tuple[float, float]]:
    if not nodes:
        return {}
    outgoing = {}
    indegree = {node.id: 0 for node in nodes}
    for edge in edges:
        outgoing.setdefault(edge.src_id, []).append(edge.dst_id)
        indegree[edge.dst_id] = indegree.get(edge.dst_id, 0) + 1
    entry = next((node.id for node in nodes if node.is_entry), None)
    roots = [entry] if entry else [node_id for node_id, count in indegree.items() if count == 0]
    if not roots:
        roots = [nodes[0].id]
    levels: dict[int, int] = {}
    queue = [(root, 0) for root in roots]
    while queue:
        node_id, level = queue.pop(0)
        if node_id in levels:
            continue
        levels[node_id] = level
        for nxt in outgoing.get(node_id, []):
            if nxt not in levels:
                queue.append((nxt, level + 1))
    next_level = max(levels.values(), default=0) + 1
    for node in nodes:
        if node.id not in levels:
            levels[node.id] = next_level
            next_level += 1
    counts: dict[int, int] = {}
    result: dict[int, tuple[float, float]] = {}
    for node in nodes:
        level = levels[node.id]
        index = counts.get(level, 0)
        counts[level] = index + 1
        result[node.id] = (70 + index * 260, 70 + level * 150)
    return result
