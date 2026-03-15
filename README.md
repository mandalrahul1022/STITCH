# STITCH: Scale-free Temporal InTervention & Contagion Harness

> A production-grade, GPU-accelerated research platform for modelling network
> contagion, quantifying intervention efficacy, and predicting outbreak
> propagation using Graph Neural Networks — containerised for reproducibility
> and HPC-ready for cluster deployment.

---

## Technical Specifications

| Dimension | Value |
|---|---|
| **Network scale** | 10⁴ nodes · 6×10⁴ directed edges |
| **Topology** | Barabási-Albert preferential attachment (m = 3) |
| **Primary hardware target** | NVIDIA RTX 50-series (CUDA 12+); auto-falls back to MPS / CPU |
| **Sparse linear algebra** | SpMV via `torch.sparse.mm` on COO tensors |
| **Eigenvalue solver** | ARPACK via `scipy.sparse.linalg.eigsh` (Lanczos iteration) |
| **Sensitivity analysis** | Saltelli quasi-random sampling · SALib Sobol decomposition |
| **GNN architecture** | 2-layer GCNConv · 705 parameters · BCEWithLogitsLoss |
| **Container runtime** | Docker 28+ (multi-stage, `python:3.11-slim`) |
| **HPC scheduler** | Slurm (job-array ready, `--dependency=afterok` chaining) |
| **Language / runtime** | Python 3.11 · PyTorch 2.10 · torch_geometric 2.7 |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     STITCH Research Platform                        │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  tensor_engine.py  —  Vectorized SEIR Core (N = 10,000)      │   │
│  │  • Sparse matrix-vector transmission  (zero Python loops)    │   │
│  │  • Phase III: async patch queue · edge rewiring · latency    │   │
│  │  • Spectral calibration: ρ(A) → λ_c pre-flight check        │   │
│  └──────────────┬───────────────────────────────────────────────┘   │
│                 │                                                   │
│       ┌─────────┴──────────┬──────────────────┐                    │
│       ▼                    ▼                  ▼                    │
│  run_pipeline.py    sensitivity_          predictive_              │
│  (Parquet + PyG)    analysis.py           model.py                 │
│  1.4 MB / run       640 Saltelli runs     2-layer GCN              │
│       │                                   AUC = 0.983              │
│       ▼                                                            │
│  data/parquet_export.py  →  data/pyg_dataset.py                   │
│  zstd columnar schema        InMemoryDataset (N, 4) features       │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Reproducible Research Infrastructure                        │   │
│  │  Dockerfile (multi-stage)  ·  docker-compose.yml             │   │
│  │  hpc/submit_pipeline · submit_sobol · submit_gnn             │   │
│  │  hpc/submit_all.sh  (Slurm --dependency=afterok chain)       │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Findings

### Finding 1 — The Epidemic is Mathematically Guaranteed to Explode

Before the first tick, the spectral radius of the adjacency matrix is computed:
- ρ(A) = 21.96 → λ_c = 0.0455
- At β = 0.40: **β / λ_c = 8.78×** (deep supercritical regime)

The simulation confirms the prediction: **9,372 / 10,000 nodes (93.7%) infected at peak**, reached at tick 14. The Susceptible population collapses to zero in under 20 ticks despite targeted patching running from tick 0.

![Epidemic Dashboard](epidemic_dashboard.png)

**What the four panels show:**
- **Top-left:** The classic SEIR curve — Susceptible (green) collapses, Infected (red) spikes to 93.7%, Patched (grey) grows slowly. The curve does not flatten.
- **Top-right:** Hub nodes (purple) ignite the outbreak first. Non-hub nodes (red) carry the bulk. 1,090 hubs exist — patching them buys time but cannot stop the cascade of 8,910 non-hub nodes.
- **Bottom-left:** Patching queue depth (blue bars) stays shallow — nodes flow through steadily. Cumulative patched (grey line) reaches 1,085 (10.8%) by tick 100.
- **Bottom-right:** Stacked fate chart — by tick 20 the network is 90%+ red. It never recovers under β = 0.40.

---

### Finding 2 — Viral Transmissibility Drives 54.8% of All Outcome Variance

640 Monte Carlo runs with Saltelli-sampled parameters. Sobol variance decomposition:

![Sobol Sensitivity Indices](sobol_indices.png)

| Parameter | S1 (alone) | ST (total incl. interactions) | Verdict |
|---|---|---|---|
| `spread_chance` (β) | 0.437 | **0.548** | Dominant driver — attacker's weapon |
| `patching_rate` | 0.211 | **0.390** | Strongest defensive lever |
| `patch_completion_prob` | 0.035 | 0.166 | Weak alone; gains power through interactions |
| `rewire_rate` | -0.012 | 0.005 | Statistically irrelevant |

**The gap between S1 and ST for `patching_rate` (0.179)** is the largest in the set. This proves that patching doesn't just act alone — it interacts with and amplifies other parameters. The benefit of patching *shrinks* as the virus gets faster.

![Parameter Interaction Heatmap](interaction_heatmap.png)

The heatmap confirms: `spread_chance × patching_rate = 0.204` is the strongest off-diagonal interaction. Network rewiring (rightmost column) is uniformly near zero — whether the network is volatile or not makes no difference to peak infection.

---

### Finding 3 — A GCN Predicts Who Gets Infected 5 Ticks Before It Happens

A 2-layer Graph Convolutional Network trained on paired simulation snapshots `(tick_T → tick_{T+5})` predicts per-node infection status using graph topology as signal.

![GNN Performance](gnn_performance.png)

| Metric | Value |
|---|---|
| **Test Accuracy** | 93.8% |
| **AUC-ROC** | **0.983** |
| **Hub Recall** | 92.8% of future infected nodes correctly flagged |

**Left panel:** Loss drops from 0.16 → 0.03. Accuracy climbs to 93.8% and plateaus. No divergence between train and test — no overfitting.

**Right panel:** ROC curve (orange) hugs the top-left corner. AUC = 0.983 vs random baseline (0.500). If you randomly pick one node that will be infected and one that won't, the model correctly identifies the infected one **98.3% of the time**.

Practical implication: given a network snapshot right now, STITCH can flag which nodes will be compromised 5 steps from now — before lateral movement reaches them.

---

## Quantified Research Claims

| Claim | Number | Verified by |
|---|---|---|
| Spectral radius predicts pandemic a-priori | ρ(A)=21.96, λ_c=0.0455 | `tensor_engine.py` |
| β exceeds threshold by | **8.78×** | `run_pipeline.py` |
| Peak infection at | **9,372 / 10,000 (93.7%)** | `epidemic_dashboard.png` |
| Outbreak peaks at | **tick 14** | `epidemic_dashboard.png` |
| Targeted patching reduces peak vs random by | **≥50%** | assertion in `tensor_engine.py` |
| Doubling patch rate reduces peak by | **32.9%** | `run_analysis.ipynb` |
| Transmissibility drives variance by | **54.8%** | `sensitivity_analysis.py` |
| Network rewiring contributes | **~0%** | `sobol_indices.png` |
| GCN predicts infections 5 ticks ahead at | **AUC 0.983** | `predictive_model.py` |
| Hub recall | **92.8%** | `gnn_performance.png` |

---

## Quick Start

```bash
pip install -r requirements.txt

# Step 1 — run simulation, build Parquet + PyG dataset
python3 run_pipeline.py

# Step 2 — Sobol sensitivity analysis (generates sobol_indices.png + interaction_heatmap.png)
python3 sensitivity_analysis.py

# Step 3 — train GNN (generates gnn_performance.png)
python3 predictive_model.py
```

---

## Reproducible Research Infrastructure

### Docker (one command, zero setup)

```bash
docker build -t stitch .

# Full suite: pipeline → Sobol → GNN (~25 min on CPU)
docker run --rm -v $(pwd)/results:/app/outputs stitch

# GPU-accelerated (NVIDIA host)
docker run --rm --gpus all -v $(pwd)/results:/app/outputs stitch

# Individual phases
docker run --rm -e RUN_MODE=pipeline -v $(pwd)/results:/app/outputs stitch
docker run --rm -e RUN_MODE=sobol   -v $(pwd)/results:/app/outputs stitch
docker run --rm -e RUN_MODE=gnn     -v $(pwd)/results:/app/outputs stitch
```

### HPC / Slurm (university cluster)

```bash
bash hpc/submit_all.sh   # chains pipeline → sobol + gnn via --dependency=afterok
```

---

## Mathematical Core

### Vectorized SEIR Transmission

```
infected_neighbors = A · x_infected      (SpMV, COO sparse float32)
P(exposure)        = 1 − (1 − β)^k       (per-node, vectorized, zero loops)
```

### Phase III Algorithmic Mechanics

| Mechanism | Detail |
|---|---|
| **Async patching queue** | Geometric drain — each queued node completes with prob `p_drain` per tick |
| **Stochastic edge rewiring** | `rewire_rate` fraction of edges randomly reconnected per tick |
| **Latency-weighted exposure** | Queued nodes transmit at 50% rate during repair |

### Spectral Threshold

```
ρ(A) = largest eigenvalue of A   [ARPACK Lanczos, never dense]
λ_c  = 1 / ρ(A)                  [epidemic threshold]
β > λ_c  →  pandemic guaranteed
```

---

## Project Structure

| File | Role | Status |
|---|---|---|
| `tensor_engine.py` | Vectorized SEIR core, spectral calibration, Phase III | Active |
| `run_pipeline.py` | Simulation → Parquet → PyG orchestrator | Active |
| `sensitivity_analysis.py` | Sobol global sensitivity, 640 runs | Active |
| `predictive_model.py` | GCN training, ROC/accuracy visualization | Active |
| `data/parquet_export.py` | zstd Parquet snapshot exporter | Active |
| `data/pyg_dataset.py` | PyG InMemoryDataset from Parquet | Active |
| `Dockerfile` + `docker-compose.yml` | Container image and service orchestration | Active |
| `hpc/submit_*.sh` | Slurm batch scripts | Active |
| `epidemic_dashboard.png` | 4-panel SEIR epidemic visualization | Artifact |
| `sobol_indices.png` | Sobol S1/ST bar chart | Artifact |
| `interaction_heatmap.png` | Pairwise S2 interaction heatmap | Artifact |
| `gnn_performance.png` | GNN training curve + ROC | Artifact |
| `models/university_network.py` | Mesa ABM (N=200, interactive) | Legacy |
| `server.py` | Mesa browser dashboard | Legacy |
| `run_analysis.ipynb` | Mesa proof experiments | Legacy |

---

## Requirements

- Python 3.11+ · packages in `requirements.txt`
- Runs on CPU (Mac/Linux), CUDA 12+, or MPS (auto-detected)

---

## Appendix: Legacy Interactive Dashboard (Mesa)

The original Mesa agent-based model (N=200) provides a live browser UI for interactive exploration. It is self-contained and independent of the tensor engine.

```bash
python3 server.py    # open http://127.0.0.1:8521
```

The dashboard provides real-time sliders for virus spread, patching rate, and outbreak size, with four live charts: SEIR time series, daily events, device-type infection curves, and the live network graph with colour-coded nodes. The Mesa model validated the core mathematical framework before scaling to N=10,000.
