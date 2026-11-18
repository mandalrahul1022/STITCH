"""Unweighted undirected topology operations for static synthetic graphs."""
from collections import deque
import hashlib

import numpy as np


def edge_hash(edges):
    # September convention: numerically sorted pairs, two contiguous int64 rows.
    pairs = np.sort(np.asarray(edges, dtype="<i8"), axis=1)
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return hashlib.sha256(np.ascontiguousarray(pairs.T).tobytes()).hexdigest()


def neighbors(nodes, edges):
    adj = [[] for _ in range(nodes)]
    for u, v in edges.tolist():
        adj[u].append(v)
        adj[v].append(u)
    return [sorted(row) for row in adj]


def betweenness(adj):
    """Brandes' all-source algorithm. Rankings use unnormalized centrality."""
    n = len(adj)
    centrality = np.zeros(n)
    for root in range(n):
        stack = []
        parents = [[] for _ in adj]
        paths = [0] * n
        distance = [-1] * n
        distance[root], paths[root] = 0, 1
        queue = deque([root])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adj[v]:
                if distance[w] < 0:
                    distance[w] = distance[v] + 1
                    queue.append(w)
                if distance[w] == distance[v] + 1:
                    paths[w] += paths[v]
                    parents[w].append(v)
        dependency = [0.0] * n
        for w in reversed(stack):
            for v in parents[w]:
                dependency[v] += paths[v] / paths[w] * (1 + dependency[w])
            if w != root:
                centrality[w] += dependency[w]
    return centrality * 0.5


def connected_region(adj, root, count):
    ordered = [root]
    seen = {root}
    for v in ordered:
        for w in adj[v]:
            if w not in seen:
                ordered.append(w)
                seen.add(w)
        if len(ordered) >= count:
            return np.array(ordered[:count])
    raise ValueError("root's connected component is too small")
