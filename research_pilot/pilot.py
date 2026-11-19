"""Synthetic first-event forecasting. All arrays stay in process on the CPU."""
from dataclasses import dataclass
import hashlib

import numpy as np


S, E, I, P = range(4)
HISTORY = 8
HORIZON = 5
PRIMARY_ORIGINS = tuple(range(13, 44, 2))
EVENT_ORIGINS = tuple(range(7, 52, 2))
TRAIN_ORIGINS = tuple(range(7, 52, 2))  # Spec v1.0.1: fit on 7..51; score and select on PRIMARY_ORIGINS
KNOWN_SCORE = -1.0  # Spec v1.0.1: observer-known hosts rank strictly below every candidate
CHANNELS = ("visible_S", "visible_E", "visible_I", "visible_P",
            "available", "age_over_32", "seen_compromised", "normalized_log_degree")


@dataclass(frozen=True)
class Process:
    beta: float
    volatility: float = 0.20
    progression: float = 0.50
    enqueue_fraction: float = 0.10
    completion: float = 0.33


def advance(state, first, queued, reference, edges, tick, process, rng):
    """Update one tick in place. Edges are undirected, unique, and loop-free."""
    active = (state == I) & (rng.random(state.size) >= process.volatility)
    weight = active * np.where(queued, 0.5, 1.0)
    u, v = edges.T
    pressure = (np.bincount(u, weights=weight[v], minlength=state.size)
                + np.bincount(v, weights=weight[u], minlength=state.size))
    risk = 1.0 - np.power(1.0 - process.beta, pressure)
    exposed = (state == S) & (rng.random(state.size) < risk)
    first[exposed] = tick
    state[exposed] = E
    state[(state == E) & (rng.random(state.size) < process.progression)] = I
    eligible = np.flatnonzero(reference & (state == I) & ~queued)
    count = round(process.enqueue_fraction * eligible.size)
    if count:
        queued[rng.choice(eligible, count, replace=False)] = True
    before_patch = state.copy(), queued.copy()
    completed = queued & (rng.random(state.size) < process.completion)
    state[completed] = P
    queued[completed] = False
    return before_patch


def simulate(nodes, edges, seed, beta, ticks=56):
    """Roll out through t=ticks inclusive, with t=0 the initial state."""
    rng = np.random.default_rng(seed)
    state = np.full(nodes, S, dtype=np.uint8)
    first = np.full(nodes, -1, dtype=np.int16)
    state[rng.choice(nodes, 3, replace=False)] = I
    first[state == I] = 0
    reference = np.zeros(nodes, dtype=bool)
    reference[rng.choice(nodes, round(0.8 * nodes), replace=False)] = True
    queue = np.zeros(nodes, dtype=bool)
    states = np.empty((ticks + 1, nodes), dtype=np.uint8)
    queues = np.empty_like(states, dtype=bool)
    states[0], queues[0] = state, queue
    process = Process(beta=beta)
    pre_state = np.empty_like(states)
    pre_queue = np.empty_like(queues)
    pre_state[0], pre_queue[0] = state, queue
    for tick in range(1, ticks + 1):
        pre_state[tick], pre_queue[tick] = advance(state, first, queue, reference, edges, tick, process, rng)
        states[tick], queues[tick] = state, queue
    return dict(states=states, first=first, queues=queues, reference=reference,
                pre_patch_state=pre_state, pre_patch_queue=pre_queue)


def truth_hash(episode):
    digest = hashlib.sha256()
    for field in ("states", "first", "queues", "reference", "pre_patch_state", "pre_patch_queue"):
        array = np.ascontiguousarray(episode[field])
        digest.update(f"{field}|{array.dtype.str}|{array.shape}|".encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def labels(first, t, horizon=HORIZON):
    return (t < first) & (first <= t + horizon)


def visible_evidence(states, availability):
    """Only permitted evidence crosses this boundary. Hidden cells use sentinel -1."""
    return np.where(availability, states, -1).astype(np.int8)


def features(evidence, normalized_log_degree, compromised_codes):
    """Causal encoding. The caller supplies its visible-evidence confirmation rule."""
    n_ticks, nodes = evidence.shape
    x = np.empty((n_ticks, nodes, len(CHANNELS)), dtype=np.float32)
    age = np.full(nodes, 32, dtype=np.int32)
    known = np.zeros(nodes, dtype=bool)
    for tick, observed in enumerate(evidence):
        available = observed >= 0
        x[tick, :, :4] = observed[:, None] == np.arange(4)
        age = np.where(available, 0, np.minimum(32, age + 1))
        known |= np.isin(observed, compromised_codes)
        x[tick, :, 4] = available
        x[tick, :, 5] = age / 32
        x[tick, :, 6] = known
        x[tick, :, 7] = normalized_log_degree
    return x


def window(encoded, t):
    if t < HISTORY - 1 or t >= len(encoded):
        raise ValueError("origin lacks a complete history")
    return encoded[t - HISTORY + 1:t + 1]


def matched_nodes(degree, bridge, rng):
    """Degree-stratified sampling without replacement; overlap is permitted."""
    values, counts = np.unique(degree[bridge], return_counts=True)
    return np.concatenate([
        rng.choice(np.flatnonzero(degree == value), size=count, replace=False)
        for value, count in zip(values, counts, strict=True)
    ])


def visibility(n_ticks, nodes, condition, rng, *, persistent_set=None, hidden=102):
    out = np.ones((n_ticks, nodes), dtype=bool)
    if condition == "full":
        return out
    if condition == "scattered":
        for tick in range(12, min(44, n_ticks)):
            out[tick, rng.choice(nodes, hidden, replace=False)] = False
        return out
    if condition == "random_persistent":
        persistent_set = rng.choice(nodes, hidden, replace=False)
    elif condition not in ("bridge_persistent", "degree_matched", "region_persistent"):
        raise ValueError(f"unknown condition {condition}")
    if persistent_set is None:
        raise ValueError("provide the graph-derived host set")
    selected = np.asarray(persistent_set)
    if (selected.shape != (hidden,) or len(np.unique(selected)) != hidden
            or selected.min() < 0 or selected.max() >= nodes):
        raise ValueError("invalid persistent host set")
    out[12:44, selected] = False
    return out
