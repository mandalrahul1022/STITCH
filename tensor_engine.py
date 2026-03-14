#!/usr/bin/env python3
"""
TensorCyberSimulation: A zero-loop, vectorized SEIR network contagion engine.
Implements production-grade CUDA-accelerated contagion dynamics using PyTorch.
Python 3.12, exhaustive type annotations with explicit tensor shapes.
"""

import torch
import torch.nn.functional as F
import networkx as nx
import scipy.sparse
import scipy.sparse.linalg
from torch import Tensor
from typing import Tuple

# Device configuration: auto-detect best available backend
def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "cpu"  # MPS has limited sparse support; CPU is safest on Mac
    return "cpu"

device: str = _select_device()

# State definitions: 0: Susceptible, 1: Exposed, 2: Infected, 3: Patched
STATE_S: int = 0
STATE_E: int = 1
STATE_I: int = 2
STATE_P: int = 3

# Simulation parameters
N: int = 10000     # number of nodes
m: int = 3         # BA graph parameter
spread_chance: float = 0.4  # transmission probability (per contact) for simulation runs
patching_rate: float = 0.10  # patching rate (fraction)
num_ticks: int = 100       # number of simulation steps per Monte Carlo run

# Set random seed explicitly for all available backends
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def assert_runtime() -> None:
    """Log the selected compute device at startup."""
    print(f"TensorCyberSimulation running on device: {device}")

def build_sparse_adj_matrix() -> Tensor:
    """
    Generate a bidirectional adjacency matrix as a torch.sparse_coo_tensor.
    
    Returns
    -------
    Tensor
        A sparse COO tensor of shape (N, N) on CUDA, coalesced.
    """
    # Generate BA graph with NetworkX
    G: nx.Graph = nx.barabasi_albert_graph(n=N, m=m, seed=42)
    # Ensure bidirectionality: for each edge, add both directions
    edges = list(G.edges())
    bidirectional_edges = edges + [(v, u) for u, v in edges]
    # Prepare indices and values for sparse matrix
    indices_list = [[u, v] for u, v in bidirectional_edges]
    if indices_list:
        indices: Tensor = torch.tensor(indices_list, dtype=torch.long).t()  # shape (2, num_edges)
    else:
        indices = torch.empty((2, 0), dtype=torch.long)
    values: Tensor = torch.ones(indices.shape[1], dtype=torch.float32)
    adj_matrix: Tensor = torch.sparse_coo_tensor(indices, values, size=(N, N), device=device).coalesce()
    return adj_matrix

def initialize_node_states() -> Tensor:
    """
    Initialize the node state vector.
    
    Returns
    -------
    Tensor
        1D tensor of shape (N,) with type torch.int8, with all nodes set to Susceptible (0).
    """
    states: Tensor = torch.full((N,), STATE_S, dtype=torch.int8, device=device)
    return states

def compute_static_hub_mask(adj_matrix: Tensor) -> Tensor:
    """
    Compute static hub mask based on raw degree centrality.
    Top 10% nodes are marked True.
    
    Parameters
    ----------
    adj_matrix : Tensor
        Sparse COO tensor of shape (N, N).
        
    Returns
    -------
    Tensor
        Boolean tensor of shape (N,) on CUDA where True indicates a hub.
    """
    # Compute degrees from sparse matrix: sum along rows (dense result)
    degrees: Tensor = torch.sparse.sum(adj_matrix, dim=1).to_dense()  # shape (N,)
    # Determine degree threshold for top 10%
    threshold: float = torch.quantile(degrees, 0.90).item()
    hub_mask: Tensor = degrees >= threshold  # shape (N,), bool tensor
    return hub_mask.to(device)

def calculate_epidemic_threshold(
    adj_matrix: Tensor,
    beta: float,
) -> Tuple[float, float, bool]:
    """
    Compute the spectral radius of the adjacency matrix and derive the
    mathematical epidemic threshold via spectral graph theory.

    The critical threshold is lambda_c = 1 / rho(A), where rho(A) is the
    largest eigenvalue (spectral radius) of the adjacency matrix.  If
    beta > lambda_c the outbreak is mathematically predicted to reach
    pandemic state on the given topology.

    Uses scipy.sparse.linalg.eigsh (Lanczos) — never converts to dense.

    Parameters
    ----------
    adj_matrix : Tensor
        Sparse COO tensor of shape (N, N), float32.
    beta : float
        Per-contact transmission probability (spread_chance).

    Returns
    -------
    Tuple[float, float, bool]
        (spectral_radius, lambda_c, is_unstable)
    """
    coalesced: Tensor = adj_matrix.coalesce()
    indices: Tensor = coalesced.indices().cpu()       # (2, E)
    values = coalesced.values().cpu().numpy()          # (E,)
    size: int = coalesced.size(0)

    sp_matrix = scipy.sparse.coo_matrix(
        (values, (indices[0].numpy(), indices[1].numpy())),
        shape=(size, size),
    ).tocsr()

    eigenvalues, _ = scipy.sparse.linalg.eigsh(
        sp_matrix.astype("float64"), k=1, which="LM",
    )
    spectral_radius: float = float(abs(eigenvalues[0]))
    lambda_c: float = 1.0 / spectral_radius if spectral_radius > 0 else float("inf")
    is_unstable: bool = beta > lambda_c

    prediction: str = "UNSTABLE (pandemic)" if is_unstable else "STABLE (contained)"
    print("=" * 60)
    print("  Spectral Graph Theory — Epidemic Calibration")
    print("=" * 60)
    print(f"  Spectral Radius  rho(A) : {spectral_radius:.4f}")
    print(f"  Epidemic Threshold l_c  : {lambda_c:.6f}")
    print(f"  Current beta             : {beta}")
    print(f"  Prediction               : {prediction}")
    print(f"  beta / l_c ratio         : {beta / lambda_c:.2f}x threshold")
    print("=" * 60)

    return spectral_radius, lambda_c, is_unstable


def rewire_edges(
    adj_matrix_base: Tensor,    # shape (N, N), sparse float32 — original graph
    rewire_rate: float,         # fraction of edges to rewire
) -> Tensor:
    """
    Create a per-tick rewired copy of the base adjacency matrix.

    For each rewired edge, the source node is kept but the destination is
    replaced with a uniformly random node.  The base matrix is never mutated.

    Returns
    -------
    Tensor
        New coalesced sparse COO tensor of shape (N, N).
    """
    coalesced: Tensor = adj_matrix_base.coalesce()
    indices: Tensor = coalesced.indices().clone()   # shape (2, E)
    values: Tensor = coalesced.values().clone()     # shape (E,)
    E: int = indices.shape[1]
    num_rewire: int = int(rewire_rate * E)
    if num_rewire == 0:
        return coalesced

    perm: Tensor = torch.randperm(E, device=indices.device)[:num_rewire]
    indices[1, perm] = torch.randint(0, indices.max().item() + 1, (num_rewire,), device=indices.device)

    rewired: Tensor = torch.sparse_coo_tensor(
        indices, values, size=coalesced.size(), device=coalesced.device
    ).coalesce()
    return rewired


def simulation_step(
    state: Tensor,                   # shape (N,), torch.int8
    adj_matrix: Tensor,              # shape (N, N), sparse float32
    spread_chance: float,
    patching_rate: float,
    patching_strategy: str,
    hub_mask: Tensor,                # shape (N,), bool tensor
    patch_queue: Tensor,             # shape (N,), bool — nodes awaiting patch completion
    volatility_rate: float = 0.20,
    patch_completion_prob: float = 0.33,
    rewire_rate: float = 0.05,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Execute one simulation step with vectorized transmission, Markovian transitions,
    and intervention based on patching strategy.

    Phase II:  Stochastic Edge Dropout via volatility_rate.
    Phase III: Stochastic edge rewiring, latency-weighted exposure for queued
               nodes, and asynchronous patching queue with geometric drain.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        (state, patch_queue, adj_matrix) — all updated for next tick.
    """
    N_local: int = state.shape[0]

    # ── Phase III-B: Stochastic Edge Rewiring ────────────────────────────
    adj_matrix = rewire_edges(adj_matrix, rewire_rate)

    # ── Transmission ─────────────────────────────────────────────────────
    infected_mask: Tensor = (state == STATE_I)  # shape (N,), bool
    infected_mask_float: Tensor = infected_mask.to(torch.float32)

    # Phase III-C: Latency-weighted exposure — queued nodes transmit at half rate
    infected_mask_float = infected_mask_float * torch.where(
        patch_queue, torch.tensor(0.5, device=device), torch.tensor(1.0, device=device)
    )

    # Phase II — Stochastic Edge Dropout
    infected_mask_dropped: Tensor = F.dropout(
        infected_mask_float, p=volatility_rate, training=True
    ) * (1.0 - volatility_rate)

    infected_neighbors: Tensor = torch.sparse.mm(
        adj_matrix, infected_mask_dropped.unsqueeze(1)
    ).squeeze(1)

    exposure_prob: Tensor = 1.0 - torch.pow(
        torch.tensor(1.0 - spread_chance, device=device, dtype=torch.float32),
        infected_neighbors,
    )
    rand_exposure: Tensor = torch.rand((N_local,), device=device, dtype=torch.float32)
    exposure_events: Tensor = (rand_exposure < exposure_prob) & (state == STATE_S)
    state = torch.where(exposure_events, torch.tensor(STATE_E, device=device, dtype=torch.int8), state)

    # ── Exposed → Infected (p = 0.5) ────────────────────────────────────
    rand_infection: Tensor = torch.rand((N_local,), device=device, dtype=torch.float32)
    infection_events: Tensor = (rand_infection < 0.5) & (state == STATE_E)
    state = torch.where(infection_events, torch.tensor(STATE_I, device=device, dtype=torch.int8), state)

    # ── Intervention: enqueue candidates ─────────────────────────────────
    if patching_rate > 0.0:
        if patching_strategy == "Targeted":
            candidate_mask: Tensor = (state == STATE_I) & hub_mask & ~patch_queue
        else:
            candidate_mask: Tensor = (state == STATE_I) & ~patch_queue
        candidate_indices: Tensor = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        num_candidates: int = candidate_indices.numel()
        num_to_patch: int = int(round(patching_rate * num_candidates))
        if num_to_patch > 0 and num_candidates > 0:
            perm: Tensor = torch.randperm(num_candidates, device=device)
            selected_indices: Tensor = candidate_indices[perm[:num_to_patch]]
            patch_queue[selected_indices] = True

    # ── Phase III-A: Geometric queue drain ───────────────────────────────
    drain_roll: Tensor = torch.rand((N_local,), device=device, dtype=torch.float32)
    drain_mask: Tensor = patch_queue & (drain_roll < patch_completion_prob)
    state[drain_mask] = STATE_P
    patch_queue[drain_mask] = False

    return state, patch_queue, adj_matrix

def run_simulation(
    patching_strategy: str,
    patching_rate: float,
    spread_chance: float,
    num_ticks: int,
    seed: int,
    patch_completion_prob: float = 0.33,
    rewire_rate: float = 0.05,
    volatility_rate: float = 0.20,
) -> int:
    """
    Run a simulation for num_ticks and return the peak infected count observed.
    
    Returns
    -------
    int
        Maximum number of infected nodes (state == 2) at any time step.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    adj_matrix_base: Tensor = build_sparse_adj_matrix()
    state: Tensor = initialize_node_states()
    patch_queue: Tensor = torch.zeros(N, dtype=torch.bool, device=device)
    initial_infected: Tensor = torch.randperm(N, device=device)[:5]
    state[initial_infected] = STATE_I
    hub_mask: Tensor = compute_static_hub_mask(adj_matrix_base)

    peak_infected: int = 0
    for _ in range(num_ticks):
        state, patch_queue, _ = simulation_step(
            state, adj_matrix_base, spread_chance, patching_rate,
            patching_strategy, hub_mask, patch_queue,
            volatility_rate=volatility_rate,
            patch_completion_prob=patch_completion_prob,
            rewire_rate=rewire_rate,
        )
        current_infected: int = int(torch.sum(state == STATE_I).item())
        if current_infected > peak_infected:
            peak_infected = current_infected
    return peak_infected

def main() -> None:
    assert_runtime()
    # Run two 100-tick Monte Carlo simulations at spread_chance 0.4 with patching rate 0.10.
    # Simulation A: Random patching strategy.
    # Simulation B: Targeted hub patching strategy.
    num_sim_ticks: int = 100
    spread: float = 0.4
    patch_rate: float = 0.10
    trials: int = 100  # Use 100 Monte Carlo trials for robust averaging.
    peaks_random: list[int] = []
    peaks_targeted: list[int] = []
    
    for trial in range(trials):
        peak_random: int = run_simulation("Random", patch_rate, spread, num_sim_ticks, seed=trial)
        peak_targeted: int = run_simulation("Targeted", patch_rate, spread, num_sim_ticks, seed=trial)
        peaks_random.append(peak_random)
        peaks_targeted.append(peak_targeted)
    
    avg_peak_random: float = float(sum(peaks_random)) / len(peaks_random)
    avg_peak_targeted: float = float(sum(peaks_targeted)) / len(peaks_targeted)
    reduction_percentage: float = ((avg_peak_random - avg_peak_targeted) / avg_peak_random) * 100.0
    
    print(f"Average peak infected (Random patching): {avg_peak_random:.2f}")
    print(f"Average peak infected (Targeted patching): {avg_peak_targeted:.2f}")
    print(f"Reduction percentage: {reduction_percentage:.2f}%")
    
    # Assertion Tripwire
    assert reduction_percentage >= 50.0, f"Math divergence. Peak reduction was {reduction_percentage}%"

if __name__ == "__main__":
    main()
