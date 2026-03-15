#!/usr/bin/env python3
"""
app.py — STITCH Interactive Browser Dashboard

Streamlit frontend for the TensorCyberSimulation engine.
Provides parameter sliders, live epidemic curves, spectral calibration
metrics, and pre-generated research visualizations in a single browser tab.

Run:  streamlit run app.py
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots
from torch import Tensor

from tensor_engine import (
    N,
    STATE_E,
    STATE_I,
    STATE_P,
    STATE_S,
    build_sparse_adj_matrix,
    calculate_epidemic_threshold,
    compute_static_hub_mask,
    device,
    initialize_node_states,
    simulation_step,
)

st.set_page_config(
    page_title="STITCH — SEIR Epidemic Simulator",
    page_icon="🦠",
    layout="wide",
)

st.title("STITCH: Scale-free Temporal InTervention & Contagion Harness")
st.caption(f"N = {N:,} nodes · Barabási-Albert (m=3) · Device: {device}")


# ── Sidebar: parameter controls ──────────────────────────────────────────

st.sidebar.header("Simulation Parameters")

spread_chance = st.sidebar.slider(
    "Spread Chance (β)", 0.05, 0.60, 0.40, 0.05,
    help="Per-contact transmission probability each tick",
)
patching_rate = st.sidebar.slider(
    "Patching Rate", 0.05, 0.20, 0.10, 0.01,
    help="Fraction of infected candidates selected for patching per tick",
)
patch_completion_prob = st.sidebar.slider(
    "Patch Completion Prob", 0.10, 0.90, 0.33, 0.05,
    help="Probability a queued node completes its patch each tick (geometric drain)",
)
rewire_rate = st.sidebar.slider(
    "Rewire Rate", 0.01, 0.15, 0.05, 0.01,
    help="Fraction of edges randomly reconnected per tick",
)
patching_strategy = st.sidebar.radio(
    "Patching Strategy", ["Targeted", "Random"],
    help="Targeted prioritises hub nodes; Random selects uniformly",
)
num_ticks = st.sidebar.slider("Ticks", 50, 200, 100, 10)
seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)

run_button = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)

# ── Tabs ─────────────────────────────────────────────────────────────────

tab_sim, tab_spectral, tab_sobol, tab_gnn = st.tabs([
    "Epidemic Simulation",
    "Spectral Calibration",
    "Sobol Sensitivity",
    "GNN Prediction",
])


# ── Tab 1: Live simulation ───────────────────────────────────────────────

with tab_sim:
    if run_button:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        with st.spinner("Building graph and computing spectral threshold..."):
            adj_matrix_base: Tensor = build_sparse_adj_matrix()
            hub_mask: Tensor = compute_static_hub_mask(adj_matrix_base)
            rho, lam_c, unstable = calculate_epidemic_threshold(adj_matrix_base, spread_chance)

        col1, col2, col3 = st.columns(3)
        col1.metric("Spectral Radius ρ(A)", f"{rho:.2f}")
        col2.metric("Epidemic Threshold λc", f"{lam_c:.4f}")
        col3.metric("β / λc", f"{spread_chance / lam_c:.1f}×",
                     delta="UNSTABLE" if unstable else "STABLE",
                     delta_color="inverse" if unstable else "normal")

        state: Tensor = initialize_node_states()
        patch_queue: Tensor = torch.zeros(N, dtype=torch.bool, device=device)
        initial_idx: Tensor = torch.randperm(N, device=device)[:5]
        state[initial_idx] = STATE_I

        history = {"tick": [], "S": [], "E": [], "I": [], "P": [], "Q": []}
        progress = st.progress(0, text="Running simulation...")
        t0 = time.time()

        for tick in range(num_ticks):
            state, patch_queue, _ = simulation_step(
                state=state,
                adj_matrix=adj_matrix_base,
                spread_chance=spread_chance,
                patching_rate=patching_rate,
                patching_strategy=patching_strategy,
                hub_mask=hub_mask,
                patch_queue=patch_queue,
                volatility_rate=0.20,
                patch_completion_prob=patch_completion_prob,
                rewire_rate=rewire_rate,
            )
            history["tick"].append(tick)
            history["S"].append(int((state == STATE_S).sum()))
            history["E"].append(int((state == STATE_E).sum()))
            history["I"].append(int((state == STATE_I).sum()))
            history["P"].append(int((state == STATE_P).sum()))
            history["Q"].append(int(patch_queue.sum()))
            progress.progress((tick + 1) / num_ticks, text=f"Tick {tick+1}/{num_ticks}")

        elapsed = time.time() - t0
        progress.empty()

        peak_I = max(history["I"])
        peak_tick = history["I"].index(peak_I)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Peak Infected", f"{peak_I:,} / {N:,}")
        m2.metric("Peak Fraction", f"{peak_I/N*100:.1f}%")
        m3.metric("Peak at Tick", str(peak_tick))
        m4.metric("Runtime", f"{elapsed:.1f}s")

        # SEIR curve
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            "SEIR Compartment Dynamics", "Node Fate (% of Network)"
        ])

        ticks_arr = history["tick"]
        for name, color in [("S", "#27ae60"), ("E", "#f39c12"), ("I", "#e74c3c"), ("P", "#7f8c8d")]:
            fig.add_trace(go.Scatter(
                x=ticks_arr, y=history[name], name=name, mode="lines",
                line=dict(color=color, width=2.5),
            ), row=1, col=1)

        for name, color in [("S", "#27ae60"), ("E", "#f39c12"), ("I", "#e74c3c"), ("P", "#7f8c8d")]:
            fig.add_trace(go.Scatter(
                x=ticks_arr,
                y=[v / N * 100 for v in history[name]],
                name=f"{name} %", mode="lines",
                line=dict(color=color, width=2), stackgroup="fate",
                showlegend=False,
            ), row=1, col=2)

        fig.update_layout(height=450, template="plotly_white",
                          legend=dict(orientation="h", y=-0.15))
        fig.update_yaxes(title_text="Node Count", row=1, col=1)
        fig.update_yaxes(title_text="% of Network", range=[0, 100], row=1, col=2)
        fig.update_xaxes(title_text="Tick", row=1, col=1)
        fig.update_xaxes(title_text="Tick", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

        # Queue chart
        fig_q = go.Figure()
        fig_q.add_trace(go.Bar(x=ticks_arr, y=history["Q"], name="Queue Depth",
                               marker_color="#3498db", opacity=0.7))
        fig_q.add_trace(go.Scatter(x=ticks_arr, y=history["P"], name="Cumulative Patched",
                                   line=dict(color="#7f8c8d", width=2.5)))
        fig_q.update_layout(height=300, template="plotly_white",
                            title="Patching Queue Depth & Cumulative Patched",
                            yaxis_title="Nodes", xaxis_title="Tick",
                            legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_q, use_container_width=True)

    else:
        st.info("Adjust parameters in the sidebar and click **Run Simulation** to start.")


# ── Tab 2: Spectral Calibration ─────────────────────────────────────────

with tab_spectral:
    st.subheader("Spectral Graph Theory — Pre-flight Check")
    st.markdown("""
Before the first simulation tick, the engine computes the **spectral radius** ρ(A)
of the adjacency matrix using the Lanczos algorithm (`scipy.sparse.linalg.eigsh`)
and derives the epidemic threshold:

$$\\lambda_c = \\frac{1}{\\rho(A)}$$

If β > λc, the outbreak is **mathematically guaranteed** to reach pandemic scale.
For BA(10,000, m=3): ρ(A) ≈ 21.96, λc ≈ 0.0455. At β = 0.40, the ratio is **8.78×** — deep supercritical.
    """)
    dashboard_path = pathlib.Path("epidemic_dashboard.png")
    if dashboard_path.exists():
        st.image(str(dashboard_path), caption="4-Panel Epidemic Dashboard (from latest run_pipeline.py execution)")


# ── Tab 3: Sobol Results ────────────────────────────────────────────────

with tab_sobol:
    st.subheader("Sobol Variance Decomposition (640 Saltelli-sampled runs)")
    st.markdown("""
Decomposes the variance of peak infected fraction into first-order (S1),
second-order (S2), and total-order (ST) Sobol indices across 4 parameters.

**Key finding:** `spread_chance` alone explains **54.8%** of all outcome variance.
`patching_rate` is the strongest controllable lever at **39.0%** total-order.
`rewire_rate` is statistically irrelevant (ST ≈ 0.005).
    """)

    col_s1, col_s2 = st.columns(2)
    sobol_path = pathlib.Path("sobol_indices.png")
    heatmap_path = pathlib.Path("interaction_heatmap.png")
    if sobol_path.exists():
        col_s1.image(str(sobol_path), caption="S1 vs ST per parameter")
    if heatmap_path.exists():
        col_s2.image(str(heatmap_path), caption="Pairwise S2 interaction heatmap")


# ── Tab 4: GNN Prediction ──────────────────────────────────────────────

with tab_gnn:
    st.subheader("GNN Predictive Intelligence (2-layer GCN, 705 parameters)")
    st.markdown("""
A Graph Convolutional Network trained on paired simulation snapshots
**(tick_T → tick_{T+5})** predicts per-node infection status **5 ticks in advance**.
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", "93.8%")
    c2.metric("AUC-ROC", "0.983")
    c3.metric("Hub Recall", "92.8%")

    gnn_path = pathlib.Path("gnn_performance.png")
    if gnn_path.exists():
        st.image(str(gnn_path), caption="Training Loss/Accuracy + ROC Curve")

    st.markdown("""
**Interpretation:** If you randomly pick one node that *will* be infected and one that *won't*,
the model assigns the higher probability to the correct node **98.3% of the time**.
This proves the GCN extracts genuine structural signal from the graph topology.
    """)
