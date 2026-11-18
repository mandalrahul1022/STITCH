# STITCH feasibility pilot rebuild

This is an independently implemented synthetic forecasting pilot against base commit 43ca102ee8de944acf2177cb915db945999d08f9. The ZIP and cumulative binary Git patch include the complete research_pilot directory and executed outputs. Apply one distribution format to a clean base checkout.

## Stage 1

The 24 archived BA and SBM graphs passed edge-list SHA-256 verification before use. The driver reads each outbreak seed directly from the September manifest using the graph, transmission setting, split and episode ID. It simulates 48 new trajectories, retains every episode, and records new truth hashes in data/manifest.json. All 48 new truth hashes differ from September's. The simulator independently draws from NumPy PCG64 and never reads archived trajectories. A separate read-only comparison after Stage 2 verified that all 48 rebuilt end-of-tick state histories, realized first-event times and reference visibility sets match the archived values. The archive used 156 for an unobserved first event and hashed the raw state and first-event arrays. The rebuild uses -1 and hashes field names, types, dimensions, states, first events, reference visibility and queue audit records. These encoding differences account for different digests even with equivalent state trajectories. The evidence does not support attributing the hash differences to different realized RNG outcomes.

Forecasts use eight ticks of observed evidence. A visible E, I or P sets channel 6 permanently. A hidden cell has zero visible-state channels. Observation age saturates at 32. Static degree uses log1p(degree) divided by log1p(maximum degree). Ground truth, the reference defender's visibility and both queue histories stay in separate episode files.

Only an infectious host observed by the fixed reference defender is eligible to enter the queue. Enqueueing occurs after latent progression. A host can complete patching in the same tick as queue entry. Episode files record the pre-completion state and queue, and the end-of-tick state and queue, so every patch has auditable evidence. The initial state is stored at t=0 and 56 updates follow. The first S-to-E event is logged before progression and patch completion.

The data directory includes each graph, all 48 truth trajectories and 192 observation views. The study manifest links each view to its shared episode truth. Test graphs receive all six masks. Training and validation graphs receive only the three familiar masks. The control samples within exact degree strata. Betweenness ties resolve in host-ID order. A seeded BFS traverses neighbors in host-ID order. Static targeted masks are shared across the two outbreaks per test graph. The intersection count with the bridge set is recorded per graph. The unique-event count in the manifest counts distinct hosts over the union of eligible extended windows.

## Commands

Run from the repository root. Place STITCH_Feasibility_Results.zip in that root when verifying or rebuilding Stage 1. To apply the patch, first use `git apply --binary STITCH_Stage2.patch` on the base checkout. The included data directory is the executed run. Use a fresh directory when regenerating.

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r research_pilot/requirements-pilot.txt
.\.venv\Scripts\python.exe research_pilot/verify.py --data research_pilot/data --archive STITCH_Feasibility_Results.zip
.\.venv\Scripts\python.exe research_pilot/stage1.py --archive STITCH_Feasibility_Results.zip --out new_stage1
.\.venv\Scripts\python.exe research_pilot/verify.py --data new_stage1 --archive STITCH_Feasibility_Results.zip
```

**Bash**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r research_pilot/requirements-pilot.txt
.venv/bin/python research_pilot/verify.py --data research_pilot/data --archive STITCH_Feasibility_Results.zip
.venv/bin/python research_pilot/stage1.py --archive STITCH_Feasibility_Results.zip --out new_stage1
.venv/bin/python research_pilot/verify.py --data new_stage1 --archive STITCH_Feasibility_Results.zip
```

The checks write data/verification.json (or verification.json under the selected data directory). Linux CPU execution is verified. Windows instructions use pathlib and the Python 3.12 launcher. These CPU computations do not depend on a PyTorch build or CUDA extension.

## Stage 2

The cumulative Stage 2 package contains the executed data, fitted classical models, per-view predictions, and results. Use the Stage 2 patch directly on the base commit. The Stage 1 patch is a separate earlier snapshot.

**Windows PowerShell** (after environment setup above)

```powershell
.\.venv\Scripts\python.exe research_pilot/verify_classical.py --data research_pilot/data --out research_pilot/classical_results
.\.venv\Scripts\python.exe research_pilot/stage2.py --data research_pilot/data --out new_stage2
.\.venv\Scripts\python.exe research_pilot/verify_classical.py --data research_pilot/data --out new_stage2
```

**Bash**

```bash
.venv/bin/python research_pilot/verify_classical.py --data research_pilot/data --out research_pilot/classical_results
.venv/bin/python research_pilot/stage2.py --data research_pilot/data --out new_stage2
.venv/bin/python research_pilot/verify_classical.py --data research_pilot/data --out new_stage2
```

Each new Stage 2 run uses a fresh output directory. The evaluator fits on train only. It saves the chosen configurations, validation AP and fitted-model file before reading the held-out test views. All model outputs have the same eight-tick input window. Channel 6 and observation age retain the history accumulated before that window. Scoring uses one prediction per model, host, origin and view. Both evaluation populations reuse those predictions. The post-run checks inspect saved outputs.

### Comparator definitions

| Method | Inputs and fitting |
|---|---|
| Prevalence | Constant first-event rate across all training host-origins at the primary time grid. |
| Degree | Training-fitted logistic mapping from normalized log-degree to event probability. |
| Observed neighbor | Training-fitted logistic mapping from the current count of visibly infectious neighbors. |
| Mask only | Logistic regression on current and eight-tick mean availability and observation age. |
| Logistic regression | Current and window-mean values of the eight channels plus neighbor sums for visible E, visible I and availability at the origin and averaged across the window. Validation chooses regularization C from 0.1 and 1. |
| Boosted trees | Histogram gradient boosting on the same features. Validation chooses 7 or 15 leaves. The fixed configuration uses 90 boosting iterations, learning rate 0.08 and L2 penalty 1. |
| Fitted mechanistic | A four-state mean-field filter. Effective exposure hazard, latent progression, removal rate and initial-state proportions are estimated from fully visible training views. The hazard fit uses the count of visible infectious neighbors. Filtering conditions on visible evidence each tick in the window. Forecast risk is the expected loss of susceptible probability over the next five ticks. The filter receives neither true hidden state, queue, reference mask, episode beta nor future observations. |

The degree and observed-neighbor comparators each use one explanatory feature. Training calibration permits Brier scoring in probability units. The three small logistic comparators use fixed C=1. Supervised models fit all training hosts under all three familiar views. Validation compares all-host AP averaged across conditions within outbreak, outbreaks within graph, and finally graphs. The population, grid, test masks and primary model family definitions are fixed before test evaluation. File validation_selection.json contains the surviving choices and the trial scores.

### How to read the results

- All-host evaluation assigns zero probability to previously confirmed hosts. Candidate-only evaluation removes them. A visible P confirms prior compromise even if E and I were never seen. Only bridge_prevalence.json and its episode-level diagnostic compare E/I versus E/I/P eligibility.
- ROC-AUC, scikit-learn average precision, Brier score, prevalence and budget scores use the 16 primary origins. Within an episode, AUC and AP pool the primary host-origin rows. At each origin, rank all eligible hosts and take up to 13. Ties use a fixed permutation seeded by the episode and origin and shared across methods and outage conditions. Precision and late-detection share use total selected alerts as denominator. Recall uses the total count of positive host-origins.
- Unique-event recall uses all 23 extended origins. Each host with a first event in the union of those forecast windows contributes once. Lead time is measured in ticks from the earliest correct alert within horizon to the event, averaged among warned events. Undefined metrics are recorded as NaN and omitted from the corresponding mean.
- Each graph receives equal weight after averaging its two outbreaks. The exploratory_intervals and exploratory_paired_differences CSV files report **exploratory** percentile intervals from 2,000 resamples of the same eight whole test graphs, using identical draw indices for all methods. The paired-difference sign is minuend minus subtrahend. These are unadjusted for multiple comparisons.
- bridge_prevalence.json records both the graph-averaged candidate prevalence used in summary.csv and the pooled positive and eligible host-origin counts. Pooling gives more weight to episodes with more eligible hosts. Degree-control overlap is a set intersection count out of 102 per graph in data/manifest.json.
- classical_summary.log gives the compact seven-method by six-condition all-host table. summary.csv includes every requested metric under both population definitions. episode_metrics.csv and graph_metrics.csv preserve the aggregation steps. The predictions directory stores arrays indexed by the extended origins in chronological order and then by host ID. The fitted models and effective process parameters are in classical_models.joblib and mechanistic_parameters.json.

### Executed diagnostic

Under bridge_persistent, graph-averaged candidate prevalence was 0.075807 with E/I confirmation and 0.076234 with E/I/P confirmation. E/I/P removed 172 guaranteed-negative host-origin rows and preserved all 3,674 positive rows. September's reported 0.0758 is the rounded comparison value. All fitted methods and primary results use E/I/P. The alternate confirmation rule is used exclusively for the diagnostic.

The read-only archive comparison ran after model selection and test scoring. Its per-episode results are in classical_results/post_run_trajectory_audit.json. To repeat it, invoke research_pilot/compare_trajectories.py with --archive STITCH_Feasibility_Results.zip --data research_pilot/data --out trajectory_comparison.json using the same Python interpreter as above.
