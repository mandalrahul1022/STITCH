# Classical model features (22)

Source: `research_pilot/classical.py::design`. For origin t, host v, and window W = ticks t-7..t (HISTORY = 8). x[τ, v, c] is observation channel c (`pilot.CHANNELS`). A is the undirected 0/1 adjacency matrix of the graph.

| # | Name | Formula |
|---|---|---|
| 0 | visible_S | x[t, v, 0]: 1 if v is visible at t and observed S |
| 1 | visible_E | x[t, v, 1] |
| 2 | visible_I | x[t, v, 2] |
| 3 | visible_P | x[t, v, 3] |
| 4 | available | x[t, v, 4]: 1 if v is visible at t |
| 5 | age_over_32 | x[t, v, 5] = min(32, ticks since v was last visible) / 32 |
| 6 | seen_compromised | x[t, v, 6]: 1 once v has been observed in E, I or P |
| 7 | normalized_log_degree | x[t, v, 7] |
| 8–15 | mean_<channel> | (1/8) Σ_{τ∈W} x[τ, v, c] for c = 0..7, in the order above |
| 16 | nbr_visible_E | Σ_u A[v,u] x[t, u, 1] |
| 17 | nbr_visible_I | Σ_u A[v,u] x[t, u, 2] |
| 18 | nbr_available | Σ_u A[v,u] x[t, u, 4] |
| 19 | nbr_mean_visible_E | Σ_u A[v,u] · mean_W x[·, u, 1] |
| 20 | nbr_mean_visible_I | Σ_u A[v,u] · mean_W x[·, u, 2] |
| 21 | nbr_mean_available | Σ_u A[v,u] · mean_W x[·, u, 4] |

Column subsets per method (`classical.columns`):
- degree: [7]
- observed_neighbor: [17]
- mask_only: [4, 5, 12, 13], which are available, age_over_32, mean_available and mean_age_over_32
- logistic and boosted_trees: all 22

Logistic regression standardises the features first (StandardScaler); boosted trees use them raw. None of the features use hidden state, the patch queue, or the tick index.

Training rows (Spec v1.0.1): origins t = 7, 9, …, 51 (23 origins) × 3 training masks × 6,144 host-episodes = 423,936 rows. Validation selection and scoring use the primary origins 13..43.
