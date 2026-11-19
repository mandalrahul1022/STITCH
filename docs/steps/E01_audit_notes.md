# E01 audit notes

Audit of the inherited `research_pilot/` code on branch `pilot-rebuild`. The code is a reimplementation from a written specification. It is not the original September pilot code, and nothing here reproduces the September 29 results.

## C02 conformance

Read-only review of the Stage 1 and 2 code against research contract v1.0, section 3. Nothing was changed. Code hash at review time: `c1f2fdd43562` (commit `13f2ae6`). Line numbers refer to files under `research_pilot/` at that commit.

Statuses: MATCH (does what section 3 says), DEVIATION (differs from section 3 or resolves an ambiguity in it), MISSING (required by section 3, absent).

### Required items

| # | Item | File:line | Status | Evidence |
|---|------|-----------|--------|----------|
| 1 | `C(v)` recorded inside the tick before later transitions | pilot.py:35 | MATCH | `first[exposed] = tick` runs before E to I (pilot.py:37) and patch completion (pilot.py:43-44) in the same tick. Stage 1 test `test_02_capture_before_progression_or_patch` and `test_08_tiny_graph_within_tick` pass. |
| 2 | Initial infected recorded with `C = 0` | pilot.py:54-55 | MATCH | Three hosts drawn without replacement are set to I, then `first[state == I] = 0`. |
| 3 | Horizon open left, closed right | pilot.py:83 | MATCH | `(t < first) & (first <= t + horizon)` with horizon 5. `test_09_horizon_edges` passes. |
| 4 | Volatility 0.20; queued sources at half weight | pilot.py:20, 28-29 | MATCH | A source is active when its uniform draw is at least 0.20 (probability 0.80); `weight = active * np.where(queued, 0.5, 1.0)`. |
| 5 | Exposure uses weighted active infectious neighbors | pilot.py:30-34 | MATCH | Neighbor weights are summed in both edge directions with `np.bincount`; `risk = 1 - (1 - beta) ** pressure`. |
| 6 | Reference defender 80% fixed per episode; queue only from reference-observed infected; `round(0.10 x eligible)`; completion 0.33 | pilot.py:38-44, 57 | DEVIATION | Reference set is drawn once per episode (pilot.py:57). Queue eligibility is `reference & (state == I) & ~queued` (pilot.py:38), so only infectious (I) hosts can enter. E hosts never enter, and S and E hosts are never patched. Section 3.1 says "S, E and I hosts can be patched"; section 3.2 says "only reference-observed infected hosts" enter the queue. The two cannot both hold for S hosts, and "infected" may or may not include E. Also unstated in section 3 and fixed by the code: order within a tick is exposure, E to I, enqueue, completion, so a host can enter the queue and complete in the same tick (1,432 of 4,114 completions); `round` is Python's half-to-even rounding (5 eligible gives 0 entries, 25 gives 2). Completion probability 0.33 matches. See note A. |
| 7 | Response history identical across masks | stage1.py:60, 71-79 | MATCH | Each episode is simulated once (stage1.py:60). Masks are applied afterwards and only to the observation views (stage1.py:72-79), so queue and patch history cannot depend on the mask. No test checks this directly yet; T10b is added in C03. |
| 8 | Channels 0 to 7 as section 3.3 | pilot.py:91-105; stage1.py:42 | MATCH | Ch 0-3: one-hot of visible state, all zero when hidden (hidden cells carry -1, pilot.py:86-89, 99). Ch 4: availability. Ch 5: age reset to 0 when visible, else `min(32, age + 1)`, divided by 32; a never-observed host starts at 32. Ch 6: cumulative OR of visible E, I or P (causal). Ch 7: `log1p(degree) / log1p(max degree)` per graph. |
| 9 | No queue, absolute tick or hidden truth in any input | pilot.py:86-89, 91-105; classical.py:24-35 | MATCH | `features` receives only visible evidence, degree and confirmation codes. The classical design matrix uses only `x` and the adjacency matrix. `test_10_input_schema_and_clock_queue_exclusion` and `test_03_hidden_truth_invariant` pass. |
| 10 | Outage `12 <= t < 44`, 102 hosts per tick | pilot.py:129, 142; stage1.py:74-77 | MATCH | Scattered hides a fresh 102 each tick in `range(12, 44)`; persistent masks set `out[12:44, selected] = False`; stage1 asserts 102 hidden per tick inside the window and 0 outside. |
| 11 | Bridge set from betweenness on the pre-outbreak graph | stage1.py:48; topology.py:23-50 | MATCH | Brandes betweenness on the static graph, computed before `simulate` runs; top 102 with stable tie-breaking by host ID. |
| 12 | Degree-matched histogram exact; overlap reported | stage1.py:49-53; pilot.py:115-121 | MATCH | Sampling within exact degree strata; `assert_array_equal` on sorted degrees; `matched_bridge_overlap` written per test graph in `manifest.json`. |
| 13 | Region set BFS-connected | stage1.py:50; topology.py:53-63 | MATCH | BFS from a seeded random root, neighbors in host-ID order, first 102 reached. Connected by construction. |
| 14 | Candidate rule uses observer-known E/I/P (channel 6), never true state | stage2.py:96, 103 | MATCH | `known = x[origins, :, 6]`; candidate population is `~known`. The E/I variant for the bridge diagnostic uses visible E and I channels only (stage2.py:110-112). |
| 15 | All-host scoring forces 0 where channel 6 = 1, for every method | classical.py:133; stage2.py:101-102 | MATCH | `np.where(x[origins, :, 6] == 1, 0, raw)` for every method, and stage2 raises if any known host has a nonzero score. |
| 16 | Alert budget 13; late share uses `C(v) <= t` | metrics.py:16, 42, 47 | MATCH | Top 13 per origin; `prior = (first >= 0) & (first <= t)` from the episode truth (evaluation only). |
| 17 | AP via `sklearn.metrics.average_precision_score` | metrics.py:44; stage2.py:47 | MATCH | Used for both test metrics and validation AP. |
| 18 | Average the two beta episodes within a graph, then across graphs | metrics.py:83-86, 91 | MATCH | Requires exactly 2 episodes per graph, takes their finite mean, then the finite mean over graphs. |
| 19 | fitted_mechanistic reads neither hidden truth nor queue | classical.py:64-121; stage2.py:59-60 | MATCH | Parameters are fitted on training views of the `full` condition only (asserted at classical.py:68). The filter at test time reads only the observation view and adjacency; it overwrites belief with visible states and zeroes S for hidden hosts with channel 6 = 1. It never opens an episode file or the queue. |
| 20 | Seed streams separate and recorded | stage1.py:47, 59, 71; metrics.py:20, 71; classical.py:51 | DEVIATION | Streams are distinct: outbreak seed from the September manifest (also drives the reference set and initial infected), mask seed `[graph, episode, condition, 503]`, structured-set seed `[graph, 918]`, tie-break seed `[outbreak seed, t, 773]`, bootstrap 1702, boosted trees 1701. Recorded in `manifest.json`: outbreak and mask seeds only. The structured-set, tie-break, bootstrap and model seeds exist only in code. |

### Additional differences from section 3

| # | Item | File:line | Status | Evidence |
|---|------|-----------|--------|----------|
| 21 | Graphs generated from seeds 1000-3007 (BA even, SBM odd) | archive_graphs.py:13-44; stage1.py:23 | DEVIATION | No BA or SBM generator exists. `verified_graphs` reads the 24 September edge lists out of `results/data.pkl` in the September zip (raw byte literals via `pickletools`, no unpickling) and keeps those whose SHA-256 matches the September manifest. Stage 1 cannot run without the zip, and C03's "generate the 24 graphs" and T10c "regenerate graph hashes from the manifest" can only be met by replay. See note B. |
| 22 | Truth hash = SHA-256 of state history plus `C` vector | pilot.py:73-79 | DEVIATION | The hash also covers queue history, the reference set and the pre-completion state and queue, with field names, dtypes and shapes in the digest. It is a superset of section 3.2 and is why all 48 truth hashes differ from September's even though the trajectories are equal (note B). |
| 23 | Bootstrap 10,000 resamples | metrics.py:71-72 | DEVIATION | 2,000 resamples, seed 1702. Section 3.5 asks for 10,000. The older rebuild prompt did not set a number. |
| 24 | Reliability (10 equal-mass bins) and ECE | metrics.py | MISSING | Brier score is computed; reliability tables and ECE are not. |
| 25 | NaN metrics recorded with a count | metrics.py:62-65, 98 | DEVIATION | Undefined metrics become NaN and are skipped by `finite_mean`. `exploratory_intervals.csv` carries `n_graphs`, but `summary.csv` and `classical_summary.log` give no count. |
| 26 | `verification.json` rows: test ID, name, PASS/FAIL, evidence | verify.py:229-232 | DEVIATION | Rows are `name: PASS/FAIL` plus audit counts. No T-IDs, no evidence strings. Tests live in `verify.py` and `verify_classical.py` (unittest), not `research_pilot/tests/` (pytest). C03 maps them to T01-T10c. |
| 27 | Tests T10b and T10c | verify.py | MISSING | No test compares truth hashes across the six conditions (T10b) or regenerates from the manifest (T10c). Added in C03. |
| 28 | Pinned requirements match the runtime | requirements-pilot.txt | DEVIATION | Pins numpy 2.3.5, scipy 1.17.0, threadpoolctl 3.6.0; the venv has 2.4.3, 1.17.1, 3.8.0 (C00). Regenerating Stage 1 and 2 under the venv gave identical Stage 1 data and Stage 2 metrics within 1.5e-6. |
| 29 | Classical model selection on validation | stage2.py:41-50, 61-75 | MATCH, note | Selection uses validation views only (held-out masks are rejected at stage2.py:26-27). The selection AP is all-host AP with known hosts at 0, not candidate AP; section 3.8 specifies candidate AP only for the neural models. |

### Notes

**A. Queue eligibility (item 6).** Under the code, a patched host must have been reference-observed in state I while unqueued; `test_11_P_only_from_reference_observed_I_in_queue` passes on all 48 episodes (4,114 completions). Because the 48 rebuilt state histories equal September's (note B), September's simulator produced the same patch transitions from the same random draws, which is consistent with the same I-only queue rule, the same in-tick order and the same rounding. This is a statement about trajectory equality only.

**B. Trajectory comparison (run in C02, read-only).** `python research_pilot/compare_trajectories.py --archive <zip> --data research_pilot/data --out <scratch>` exited 0. Over 48 episodes: archived hash verified 48, state histories equal 48, realized first-event times equal 48, reference sets equal 48, new truth hash differs 48. The rebuilt hashes differ because of the definition in item 22 and the never-event code (-1 here, 156 in the archive), not because of different trajectories.

### Proposed fixes (none applied; Rahul decides)

| Item | Proposal |
|------|----------|
| 6 | Keep I-only queue entry, the in-tick order and Python rounding, and amend section 3.1/3.2 wording in the methods docs to state them. Changing to E-or-I entry would change every trajectory and break item B. |
| 20 | Add a `seeds` block to `manifest.json` config naming every stream (structured-set 918, tie-break 773, bootstrap, model seeds). Metadata only; no number changes. |
| 21 | Accept archive replay as "the archived-graph decision", documented in `docs/methods/simulator.md`. Alternative: write BA/SBM generators, which would produce new graphs and break comparability with September. |
| 22 | Keep the richer hash; document it. Optionally also record a section-3.2-style hash (states + `C`) alongside. |
| 23 | Raise to 10,000 resamples. Changes interval endpoints only, not point estimates; requires rerunning Stage 2 outputs. |
| 24 | Add 10-bin equal-mass reliability and ECE to `metrics.py`. Adds columns; requires rerunning Stage 2. |
| 25 | Add a finite-count column per metric to `summary.csv`. |
| 26, 27 | Do in C03 as planned: pytest files under `research_pilot/tests/` with T-IDs, evidence strings, T10b and T10c. |
| 28 | Either repin to the venv versions or install the pinned versions. Repinning is the lighter option given the measured 1.5e-6 effect. |

### Decisions (Rahul, November 18, 2026: all proposals approved)

| Item | Decision | Where |
|------|----------|-------|
| 6 | Keep queue entry from reference-observed I hosts only, the in-tick order (exposure, E to I, enqueue, completion) and half-to-even rounding. Contract wording to be stated in `docs/methods/simulator.md` (C15). | no code change |
| 20 | Seed streams recorded in `manifest.json` (`config.seeds`) and `validation_selection.json` (`seeds`). | commit 3f8bb4d |
| 21 | Archived-graph decision: the 24 September edge lists are replayed and hash-verified; no graph generator for Track A. To be documented in `docs/methods/simulator.md` (C15). | no code change |
| 22 | Keep the extended truth hash; document it (C15). | no code change |
| 23, 24, 25 | Bootstrap 10,000 resamples; per-episode ECE and `reliability.csv` (10 equal-mass bins); `n_<metric>` counts in `summary.csv`. | commit 11a959a |
| 26, 27 | Handled in C03. | C03 |
| 28 | `requirements-pilot.txt` repinned to the `.venv` versions, plus pytest 9.2.0 (installed for the C03 test suite). | commit 7008855 |

## C03 notes

### Fizzle definition (Rahul, November 18, 2026; carried into the Phase 2 preregistration)

Each episode record in `manifest.json` carries two counts:

- (a) `hosts_compromised_beyond_seeds`: hosts with `C(v) > 0`, so the 3 initial infected hosts are excluded.
- (b) `primary_window_positives`: positive host-origins `y(v,t) = 1` summed over the primary origins `t = 13, 15, ..., 43`. The record also carries `primary_window_unique_events`, the number of distinct hosts behind those positives.

An episode is an **evaluation fizzle** when (b) = 0 (`evaluation_fizzle` in the manifest). Every episode is kept regardless.

Results on the C03 data (`research_pilot/results/stage1/run_a`, identical in `run_b`):

| Count | Min (episode) | Median | Max |
|-------|---------------|--------|-----|
| (a) hosts compromised beyond the seeds | 24 (1003_0) | 126.5 | 230 |
| (b) primary-window positive host-origins | 29 (1003_0) | 192.5 | 429 |
| distinct hosts behind (b) | 12 (1003_0) | 87 | 180 |

Evaluation fizzles: 0 of 48.

Hosts ever compromised per episode, including the 3 seeds (`ever_compromised`), by transmission setting:

| beta | Episodes | Min | Median | Max |
|------|----------|-----|--------|-----|
| 0.04 | 24 | 27 | 58.5 | 130 |
| 0.09 | 24 | 129 | 199 | 233 |

### T10b replaces "truth hash identical across masks"

Section 4.1 defines T10b as "truth hash identical across all six mask conditions for every episode". In this pipeline an episode is simulated once and stored once (`stage1.py:60`). The mask conditions are observation views computed from that one stored file, so there is no per-condition truth to hash. Comparing the hash per condition would compare a file with itself and could not fail.

The implemented T10b tests the property the spec check protects, that every condition observes the same outbreak, more directly:

1. It recomputes each stored episode's truth hash and requires it to equal the manifest.
2. For each of the 192 views, it regenerates the mask from the view's recorded `mask_seed` (and the graph's stored bridge, matched or region set).
3. It rebuilds the eight feature channels from the episode's stored truth under that mask and requires bit equality with the stored view.
4. It requires each test episode to have exactly its six conditions, and each train and validation episode exactly the three training conditions.

This is stronger in two ways. A view built from any other trajectory differs in at least one visible cell, so the check fails. It also pins each mask to its recorded seed, which a hash comparison does not.

The one difference it cannot detect is a trajectory that differs only in cells hidden in that view. Those cells cannot reach the inputs (T03). Labels and the reference defender's queue history are read only from the single stored episode file, which every condition shares by episode ID (`stage2.py`, `views`). A negative control (one flipped availability cell in a copy of the data) made T06, T10 and T10b fail.

## C04 traces

Produced by `research_pilot/tools/trace_host.py` on the C03 data. Every value below is printed by the tool. The tool decodes each compact channel cell back to the stored floats and stops if the encoding is lossy. True state, queue and reference membership are shown for audit only; they are not model inputs.

Commands:

```powershell
.\.venv\Scripts\python.exe -m research_pilot.tools.trace_host --data research_pilot\results\stage1\run_a --episode 3000_0 --host 7 --origins 11,13,15,17,19
.\.venv\Scripts\python.exe -m research_pilot.tools.trace_host --fixture t08
```

Host selection rule (fixed before looking at the trace): the first test episode in ID order that has a bridge-set host with `C(v) = 18`, then the lowest such host ID. `C = 18` puts the event inside the outage, and with odd origins the label must be 1 at t = 13, 15, 17 and 0 at t = 11 and 19. Result: episode 3000_0, host 7. This host is also in the degree-matched and region sets of graph 3000, so it is hidden for the whole outage under all three held-out masks.

Cell format: `<visible state or -> a<age> k<channel 6>`; channel 7 is constant per host and printed once.

﻿### Episode 3000_0, host 7

- split test, graph 3000 (BA), beta 0.04, outbreak seed 303777
- degree 16; channel 7 (normalized log-degree) = 0.7906
- C(v) = 18; reference-observed (NOT AN INPUT): True
- member of held-out host sets: bridge_persistent, degree_matched, region_persistent
- fitted_mechanistic (fitted on 24 training full views): hazard 0.05122, progress 0.49573, remove 0.04975

| tick | true | queue (NOT AN INPUT) | full | scattered | random_persistent | bridge_persistent | degree_matched | region_persistent |
|---|---|---|---|---|---|---|---|---|
| 0 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 |
| 1 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 |
| 2 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 |
| 3 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 |
| 4 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 |
| 5 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 |
| 6 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 |
| 7 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 |
| 8 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 |
| 9 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 |
| 10 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 |
| 11 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 | S a00 k0 |
| 12 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | - a01 k0 | - a01 k0 | - a01 k0 |
| 13 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | - a02 k0 | - a02 k0 | - a02 k0 |
| 14 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | - a03 k0 | - a03 k0 | - a03 k0 |
| 15 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | - a04 k0 | - a04 k0 | - a04 k0 |
| 16 | S | 0 | S a00 k0 | S a00 k0 | S a00 k0 | - a05 k0 | - a05 k0 | - a05 k0 |
| 17 | S | 0 | S a00 k0 | - a01 k0 | S a00 k0 | - a06 k0 | - a06 k0 | - a06 k0 |
| 18 | E | 0 | E a00 k1 | - a02 k0 | E a00 k1 | - a07 k0 | - a07 k0 | - a07 k0 |
| 19 | I | 1 | I a00 k1 | I a00 k1 | I a00 k1 | - a08 k0 | - a08 k0 | - a08 k0 |
| 20 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | - a09 k0 | - a09 k0 | - a09 k0 |
| 21 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | - a10 k0 | - a10 k0 | - a10 k0 |
| 22 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | - a11 k0 | - a11 k0 | - a11 k0 |
| 23 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | - a12 k0 | - a12 k0 | - a12 k0 |
| 24 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | - a13 k0 | - a13 k0 | - a13 k0 |
| 25 | P | 0 | P a00 k1 | - a01 k1 | P a00 k1 | - a14 k0 | - a14 k0 | - a14 k0 |
| 26 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | - a15 k0 | - a15 k0 | - a15 k0 |
| 27 | P | 0 | P a00 k1 | - a01 k1 | P a00 k1 | - a16 k0 | - a16 k0 | - a16 k0 |
| 28 | P | 0 | P a00 k1 | - a02 k1 | P a00 k1 | - a17 k0 | - a17 k0 | - a17 k0 |
| 29 | P | 0 | P a00 k1 | - a03 k1 | P a00 k1 | - a18 k0 | - a18 k0 | - a18 k0 |
| 30 | P | 0 | P a00 k1 | - a04 k1 | P a00 k1 | - a19 k0 | - a19 k0 | - a19 k0 |
| 31 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | - a20 k0 | - a20 k0 | - a20 k0 |
| 32 | P | 0 | P a00 k1 | - a01 k1 | P a00 k1 | - a21 k0 | - a21 k0 | - a21 k0 |
| 33 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | - a22 k0 | - a22 k0 | - a22 k0 |
| 34 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | - a23 k0 | - a23 k0 | - a23 k0 |
| 35 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | - a24 k0 | - a24 k0 | - a24 k0 |
| 36 | P | 0 | P a00 k1 | - a01 k1 | P a00 k1 | - a25 k0 | - a25 k0 | - a25 k0 |
| 37 | P | 0 | P a00 k1 | - a02 k1 | P a00 k1 | - a26 k0 | - a26 k0 | - a26 k0 |
| 38 | P | 0 | P a00 k1 | - a03 k1 | P a00 k1 | - a27 k0 | - a27 k0 | - a27 k0 |
| 39 | P | 0 | P a00 k1 | - a04 k1 | P a00 k1 | - a28 k0 | - a28 k0 | - a28 k0 |
| 40 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | - a29 k0 | - a29 k0 | - a29 k0 |
| 41 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | - a30 k0 | - a30 k0 | - a30 k0 |
| 42 | P | 0 | P a00 k1 | - a01 k1 | P a00 k1 | - a31 k0 | - a31 k0 | - a31 k0 |
| 43 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | - a32 k0 | - a32 k0 | - a32 k0 |
| 44 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 |
| 45 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 |
| 46 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 |
| 47 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 |
| 48 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 |
| 49 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 |
| 50 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 |
| 51 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 |
| 52 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 |
| 53 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 |
| 54 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 |
| 55 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 |
| 56 | P | 0 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 | P a00 k1 |

| origin t | y(v,t) | full: status, score | scattered: status, score | random_persistent: status, score | bridge_persistent: status, score | degree_matched: status, score | region_persistent: status, score |
|---|---|---|---|---|---|---|---|
| 11 | 0 | candidate, 0.0612 | candidate, 0.0612 | candidate, 0.0612 | candidate, 0.0612 | candidate, 0.0612 | candidate, 0.0612 |
| 13 | 1 | candidate, 0.0612 | candidate, 0.0791 | candidate, 0.0988 | candidate, 0.1308 | candidate, 0.1318 | candidate, 0.1312 |
| 15 | 1 | candidate, 0.0612 | candidate, 0.1218 | candidate, 0.1359 | candidate, 0.1993 | candidate, 0.2001 | candidate, 0.1987 |
| 17 | 1 | candidate, 0.2887 | candidate, 0.2948 | candidate, 0.3611 | candidate, 0.2559 | candidate, 0.2554 | candidate, 0.2519 |
| 19 | 0 | known, 0.0000 | known, 0.0000 | known, 0.0000 | candidate, 0.2524 | candidate, 0.2501 | candidate, 0.2544 |

| check | result | detail |
|---|---|---|
| label is 1 exactly for C-5 <= t <= C-1 | PASS | C=18: y=1 for t=13..17, 0 at t=12 and t=18 |
| full: hidden ticks have channels 0-4 all 0 | PASS | 0 hidden ticks |
| full: channel 6 turns on only at a visible E/I/P tick | PASS | onset tick [18] |
| scattered: hidden ticks have channels 0-4 all 0 | PASS | 13 hidden ticks |
| scattered: channel 6 turns on only at a visible E/I/P tick | PASS | onset tick [19] |
| random_persistent: hidden ticks have channels 0-4 all 0 | PASS | 0 hidden ticks |
| random_persistent: channel 6 turns on only at a visible E/I/P tick | PASS | onset tick [18] |
| bridge_persistent: hidden ticks have channels 0-4 all 0 | PASS | 32 hidden ticks |
| bridge_persistent: channel 6 turns on only at a visible E/I/P tick | PASS | onset tick [44] |
| degree_matched: hidden ticks have channels 0-4 all 0 | PASS | 32 hidden ticks |
| degree_matched: channel 6 turns on only at a visible E/I/P tick | PASS | onset tick [44] |
| region_persistent: hidden ticks have channels 0-4 all 0 | PASS | 32 hidden ticks |
| region_persistent: channel 6 turns on only at a visible E/I/P tick | PASS | onset tick [44] |

What the episode trace shows:

- The label is 1 at origins 13, 15 and 17 and 0 at 11 and 19, and the full check confirms y = 1 exactly for t = 13..17 (C-5 to C-1).
- Under bridge_persistent the host is compromised at tick 18, queued at 19 and patched at 20, all while hidden. Channel 6 stays 0 until the host is visible again at tick 44, so at origin 19 it is still a candidate. Its score there (0.2524) counts as a late alert in the metrics, because `C(v) <= t`.
- Under full and random_persistent the observer sees E at tick 18, channel 6 turns on, and at origin 19 the score is forced to 0 (known host).
- Under scattered the host is hidden at ticks 17 and 18 and first seen compromised as I at 19.
- At origin 11 the window (ticks 4 to 11) precedes the outage, so all six conditions give the same score (0.0612).

﻿### T08 fixture, host 1

- 2 hosts, one edge; host 0 is I from tick 0 and not reference-observed; host 1 is reference-observed. Ticks 1-3 use beta 0; from tick 4, beta 1, volatility 0, progression 1, enqueue fraction 1, completion 1.
- C(v) = 4; channel 7 = 1.0 (both hosts have degree 1)
- Second view hides host 1 at ticks 3-5 to show channel 6 waiting for a visible tick.
- No comparator score: the fixture has no fitted model.

| tick | true | queue (NOT AN INPUT) | full | host1_hidden_t3_5 |
|---|---|---|---|---|
| 0 | S | 0 | S a00 k0 | S a00 k0 |
| 1 | S | 0 | S a00 k0 | S a00 k0 |
| 2 | S | 0 | S a00 k0 | S a00 k0 |
| 3 | S | 0 | S a00 k0 | - a01 k0 |
| 4 | P | 0 | P a00 k1 | - a02 k0 |
| 5 | P | 0 | P a00 k1 | - a03 k0 |
| 6 | P | 0 | P a00 k1 | P a00 k1 |
| 7 | P | 0 | P a00 k1 | P a00 k1 |
| 8 | P | 0 | P a00 k1 | P a00 k1 |
| 9 | P | 0 | P a00 k1 | P a00 k1 |
| 10 | P | 0 | P a00 k1 | P a00 k1 |

| origin t | y(v,t) | full: status | host1_hidden_t3_5: status |
|---|---|---|---|
| 0 | 1 | candidate | candidate |
| 1 | 1 | candidate | candidate |
| 2 | 1 | candidate | candidate |
| 3 | 1 | candidate | candidate |
| 4 | 0 | known | candidate |
| 5 | 0 | known | candidate |

| check | result | detail |
|---|---|---|
| label is 1 exactly for C-5 <= t <= C-1 | PASS | C=4: y=1 for t=0..3, 0 at t=n/a and t=4 |
| full: hidden ticks have channels 0-4 all 0 | PASS | 0 hidden ticks |
| full: channel 6 turns on only at a visible E/I/P tick | PASS | onset tick [4] |
| host1_hidden_t3_5: hidden ticks have channels 0-4 all 0 | PASS | 3 hidden ticks |
| host1_hidden_t3_5: channel 6 turns on only at a visible E/I/P tick | PASS | onset tick [6] |

What the fixture trace shows:

- Host 1 goes S to E to I to P inside tick 4. The end-of-tick state history never shows E or I, but `C(v) = 4` is recorded, so the event survives (T02, T08).
- The queue flag is 0 at every end of tick because the host entered the queue and completed in the same tick.
- With host 1 hidden at ticks 3 to 5, channel 6 waits until tick 6, the first visible tick, and host 1 stays a candidate at origins 4 and 5.
## C05 Step A

### Verification
- Stage 1 T01: PASS
- Stage 1 T02: PASS
- Stage 1 T03: PASS
- Stage 1 T04: PASS
- Stage 1 T05: PASS
- Stage 1 T06: PASS
- Stage 1 T07: PASS
- Stage 1 T08: PASS
- Stage 1 T09: PASS
- Stage 1 T10: PASS
- Stage 1 T10b: PASS
- Stage 1 T10c: PASS
- Stage 1 T10d: PASS
- Stage 2 test_01_future_event_and_earliest_positive_lead: PASS
- Stage 2 test_02_candidate_filter_ties_and_budget: PASS
- Stage 2 test_03_mask_only_and_causal_model_inputs: PASS
- Stage 2 test_04_filter_mass_and_observed_state: PASS
- Stage 2 test_05_frozen_splits_and_complete_score_grid: PASS
- Stage 2 test_06_all_saved_scores_respect_confirmation: PASS
- Stage 2 test_07_bridge_diagnostic_and_primary_population: PASS

### bridge_persistent candidate prevalence (September 0.0758)

| rule | graph_macro | pooled | positives | eligible |
|---|---|---|---|---|
| E/I | 0.0758 | 0.0690 | 3674 | 53284 |
| E/I/P | 0.0762 | 0.0692 | 3674 | 53112 |

### Population: all

| model | condition | auc new / Sep / Δ | ap new / Sep / Δ | recall_at_budget new / Sep / Δ | late_detection_share new / Sep / Δ | flag |
|---|---|---|---|---|---|---|
| boosted_trees | bridge_persistent | 0.7284 / 0.7460 / -0.0176 | 0.1489 / 0.1477 / +0.0012 | 0.1872 / 0.2273 / -0.0401 | 0.2275 / 0.2076 / +0.0199 |  |
| boosted_trees | degree_matched | 0.7296 / 0.7544 / -0.0248 | 0.1446 / 0.1489 / -0.0043 | 0.1877 / 0.2244 / -0.0367 | 0.2175 / 0.2076 / +0.0099 |  |
| boosted_trees | full | 0.9027 / 0.9063 / -0.0036 | 0.3024 / 0.3035 / -0.0011 | 0.3253 / 0.3820 / -0.0567 | 0.0000 / 0.0000 / +0.0000 |  |
| boosted_trees | random_persistent | 0.8295 / 0.8328 / -0.0033 | 0.2237 / 0.2247 / -0.0010 | 0.2757 / 0.3031 / -0.0274 | 0.0376 / 0.0394 / -0.0018 |  |
| boosted_trees | region_persistent | 0.7830 / 0.7927 / -0.0097 | 0.1936 / 0.1941 / -0.0005 | 0.2462 / 0.2977 / -0.0515 | 0.0763 / 0.0559 / +0.0204 |  |
| boosted_trees | scattered | 0.8874 / 0.8809 / +0.0065 | 0.2729 / 0.2570 / +0.0159 | 0.3092 / 0.3278 / -0.0186 | 0.0406 / 0.0352 / +0.0054 |  |
| degree | bridge_persistent | 0.6484 / 0.6485 / -0.0001 | 0.0837 / 0.0837 / -0.0000 | 0.1024 / 0.1120 / -0.0096 | 0.4020 / 0.3975 / +0.0045 |  |
| degree | degree_matched | 0.6481 / 0.6503 / -0.0022 | 0.0834 / 0.0840 / -0.0006 | 0.1024 / 0.1119 / -0.0095 | 0.4023 / 0.4023 / +0.0000 |  |
| degree | full | 0.7270 / 0.7270 / +0.0000 | 0.1379 / 0.1379 / +0.0000 | 0.1615 / 0.1925 / -0.0310 | 0.0000 / 0.0000 / +0.0000 |  |
| degree | random_persistent | 0.6816 / 0.6819 / -0.0003 | 0.1013 / 0.1017 / -0.0004 | 0.1270 / 0.1426 / -0.0156 | 0.2323 / 0.2533 / -0.0210 |  |
| degree | region_persistent | 0.6681 / 0.6684 / -0.0003 | 0.0917 / 0.0909 / +0.0008 | 0.1118 / 0.1199 / -0.0081 | 0.3462 / 0.3642 / -0.0180 |  |
| degree | scattered | 0.7227 / 0.7229 / -0.0002 | 0.1333 / 0.1329 / +0.0004 | 0.1560 / 0.1888 / -0.0328 | 0.0282 / 0.0270 / +0.0012 |  |
| fitted_mechanistic | bridge_persistent | 0.7895 / 0.7886 / +0.0009 | 0.1714 / 0.1767 / -0.0053 | 0.2058 / 0.2224 / -0.0166 | 0.1508 / 0.1307 / +0.0201 |  |
| fitted_mechanistic | degree_matched | 0.7964 / 0.7992 / -0.0028 | 0.1711 / 0.1755 / -0.0044 | 0.1856 / 0.2236 / -0.0380 | 0.1484 / 0.1220 / +0.0264 |  |
| fitted_mechanistic | full | 0.9146 / 0.9146 / +0.0000 | 0.3135 / 0.3135 / +0.0000 | 0.3295 / 0.3674 / -0.0379 | 0.0000 / 0.0000 / +0.0000 |  |
| fitted_mechanistic | random_persistent | 0.8604 / 0.8525 / +0.0079 | 0.2352 / 0.2383 / -0.0031 | 0.2798 / 0.2933 / -0.0135 | 0.0279 / 0.0303 / -0.0024 |  |
| fitted_mechanistic | region_persistent | 0.8236 / 0.8256 / -0.0020 | 0.2038 / 0.2094 / -0.0056 | 0.2358 / 0.2881 / -0.0523 | 0.0730 / 0.0463 / +0.0267 |  |
| fitted_mechanistic | scattered | 0.9057 / 0.9058 / -0.0001 | 0.2940 / 0.2924 / +0.0016 | 0.3242 / 0.3667 / -0.0425 | 0.0370 / 0.0334 / +0.0036 |  |
| logistic | bridge_persistent | 0.6496 / 0.7197 / -0.0701 | 0.1117 / 0.1280 / -0.0163 | 0.1649 / 0.1989 / -0.0340 | 0.3837 / 0.3032 / +0.0805 | auc -0.0701 |
| logistic | degree_matched | 0.6536 / 0.7388 / -0.0852 | 0.1068 / 0.1220 / -0.0152 | 0.1504 / 0.2021 / -0.0517 | 0.3918 / 0.2987 / +0.0931 | auc -0.0852 |
| logistic | full | 0.8913 / 0.9030 / -0.0117 | 0.2975 / 0.2889 / +0.0086 | 0.3250 / 0.3737 / -0.0487 | 0.0000 / 0.0000 / +0.0000 |  |
| logistic | random_persistent | 0.7909 / 0.8062 / -0.0153 | 0.1765 / 0.1833 / -0.0068 | 0.2337 / 0.2614 / -0.0277 | 0.2542 / 0.2118 / +0.0424 |  |
| logistic | region_persistent | 0.6850 / 0.7654 / -0.0804 | 0.1497 / 0.1599 / -0.0102 | 0.1952 / 0.2546 / -0.0594 | 0.2127 / 0.2166 / -0.0039 | auc -0.0804 |
| logistic | scattered | 0.8516 / 0.8775 / -0.0259 | 0.2509 / 0.2413 / +0.0096 | 0.2954 / 0.3204 / -0.0250 | 0.0418 / 0.0343 / +0.0075 |  |
| mask_only | bridge_persistent | 0.6088 / 0.6199 / -0.0111 | 0.0836 / 0.0865 / -0.0029 | 0.0725 / 0.0611 / +0.0114 | 0.1016 / 0.2169 / -0.1153 |  |
| mask_only | degree_matched | 0.6024 / 0.6151 / -0.0127 | 0.0844 / 0.0869 / -0.0025 | 0.0700 / 0.0687 / +0.0013 | 0.1055 / 0.2112 / -0.1057 |  |
| mask_only | full | 0.6543 / 0.6543 / +0.0000 | 0.0973 / 0.0973 / -0.0000 | 0.0743 / 0.1484 / -0.0741 | 0.0000 / 0.0000 / +0.0000 |  |
| mask_only | random_persistent | 0.6004 / 0.5971 / +0.0033 | 0.0822 / 0.0748 / +0.0074 | 0.0642 / 0.0375 / +0.0267 | 0.0793 / 0.1608 / -0.0815 |  |
| mask_only | region_persistent | 0.6238 / 0.6205 / +0.0033 | 0.0850 / 0.0819 / +0.0031 | 0.0810 / 0.0703 / +0.0107 | 0.0968 / 0.2767 / -0.1799 |  |
| mask_only | scattered | 0.6502 / 0.6458 / +0.0044 | 0.0960 / 0.0955 / +0.0005 | 0.0780 / 0.0706 / +0.0074 | 0.0592 / 0.0325 / +0.0267 |  |
| observed_neighbor | bridge_persistent | 0.6577 / 0.5815 / +0.0762 | 0.0942 / 0.0791 / +0.0151 | 0.1528 / 0.1690 / -0.0162 | 0.4534 / 0.4787 / -0.0253 | auc +0.0762 |
| observed_neighbor | degree_matched | 0.6534 / 0.6007 / +0.0527 | 0.0912 / 0.0845 / +0.0067 | 0.1393 / 0.1864 / -0.0471 | 0.4546 / 0.4799 / -0.0253 | auc +0.0527 |
| observed_neighbor | full | 0.8885 / 0.8872 / +0.0013 | 0.2630 / 0.2741 / -0.0111 | 0.3270 / 0.3789 / -0.0519 | 0.0000 / 0.0000 / +0.0000 |  |
| observed_neighbor | random_persistent | 0.7696 / 0.7214 / +0.0482 | 0.1398 / 0.1293 / +0.0105 | 0.2184 / 0.2388 / -0.0204 | 0.3555 / 0.3675 / -0.0120 | auc +0.0482 |
| observed_neighbor | region_persistent | 0.6910 / 0.6574 / +0.0336 | 0.1156 / 0.1144 / +0.0012 | 0.1764 / 0.2270 / -0.0506 | 0.2909 / 0.3233 / -0.0324 | auc +0.0336 |
| observed_neighbor | scattered | 0.8114 / 0.7540 / +0.0574 | 0.2009 / 0.1859 / +0.0150 | 0.2851 / 0.3173 / -0.0322 | 0.0451 / 0.0409 / +0.0042 | auc +0.0574 |
| prevalence | bridge_persistent | 0.6023 / 0.6010 / +0.0013 | 0.0762 / 0.0758 / +0.0004 | 0.0607 / 0.0883 / -0.0276 | 0.1457 / 0.3131 / -0.1674 |  |
| prevalence | degree_matched | 0.6018 / 0.6028 / -0.0010 | 0.0760 / 0.0763 / -0.0003 | 0.0613 / 0.0856 / -0.0243 | 0.1448 / 0.3263 / -0.1815 |  |
| prevalence | full | 0.6543 / 0.6543 / +0.0000 | 0.0973 / 0.0973 / -0.0000 | 0.0743 / 0.1484 / -0.0741 | 0.0000 / 0.0000 / +0.0000 |  |
| prevalence | random_persistent | 0.6101 / 0.6104 / -0.0003 | 0.0775 / 0.0781 / -0.0006 | 0.0613 / 0.1038 / -0.0425 | 0.1265 / 0.2272 / -0.1007 |  |
| prevalence | region_persistent | 0.6046 / 0.6049 / -0.0003 | 0.0775 / 0.0767 / +0.0008 | 0.0600 / 0.0904 / -0.0304 | 0.1391 / 0.3194 / -0.1803 |  |
| prevalence | scattered | 0.6500 / 0.6503 / -0.0003 | 0.0951 / 0.0953 / -0.0002 | 0.0729 / 0.1463 / -0.0734 | 0.0150 / 0.0228 / -0.0078 |  |

Flagged cells (|ΔAP|>0.02 or |ΔAUC|>0.03): 8
- logistic / bridge_persistent: auc -0.0701
- logistic / degree_matched: auc -0.0852
- logistic / region_persistent: auc -0.0804
- observed_neighbor / bridge_persistent: auc +0.0762
- observed_neighbor / degree_matched: auc +0.0527
- observed_neighbor / random_persistent: auc +0.0482
- observed_neighbor / region_persistent: auc +0.0336
- observed_neighbor / scattered: auc +0.0574

### Population: candidate

| model | condition | auc new / Sep / Δ | ap new / Sep / Δ | recall_at_budget new / Sep / Δ | late_detection_share new / Sep / Δ | flag |
|---|---|---|---|---|---|---|
| boosted_trees | bridge_persistent | 0.6532 / 0.7460 / -0.0928 | 0.1489 / 0.1477 / +0.0012 | 0.1872 / 0.2273 / -0.0401 | 0.2275 / 0.2076 / +0.0199 | auc -0.0928 |
| boosted_trees | degree_matched | 0.6560 / 0.7544 / -0.0984 | 0.1446 / 0.1489 / -0.0043 | 0.1877 / 0.2244 / -0.0367 | 0.2175 / 0.2076 / +0.0099 | auc -0.0984 |
| boosted_trees | full | 0.8428 / 0.9063 / -0.0635 | 0.3024 / 0.3035 / -0.0011 | 0.3253 / 0.3820 / -0.0567 | 0.0000 / 0.0000 / +0.0000 | auc -0.0635 |
| boosted_trees | random_persistent | 0.7740 / 0.8328 / -0.0588 | 0.2237 / 0.2247 / -0.0010 | 0.2757 / 0.3031 / -0.0274 | 0.0376 / 0.0394 / -0.0018 | auc -0.0588 |
| boosted_trees | region_persistent | 0.7204 / 0.7927 / -0.0723 | 0.1936 / 0.1941 / -0.0005 | 0.2462 / 0.2977 / -0.0515 | 0.0763 / 0.0559 / +0.0204 | auc -0.0723 |
| boosted_trees | scattered | 0.8211 / 0.8809 / -0.0598 | 0.2729 / 0.2570 / +0.0159 | 0.3092 / 0.3278 / -0.0186 | 0.0406 / 0.0352 / +0.0054 | auc -0.0598 |
| degree | bridge_persistent | 0.5479 / 0.6485 / -0.1006 | 0.0837 / 0.0837 / -0.0000 | 0.1024 / 0.1120 / -0.0096 | 0.4020 / 0.3975 / +0.0045 | auc -0.1006 |
| degree | degree_matched | 0.5482 / 0.6503 / -0.1021 | 0.0834 / 0.0840 / -0.0006 | 0.1024 / 0.1119 / -0.0095 | 0.4023 / 0.4023 / +0.0000 | auc -0.1021 |
| degree | full | 0.6016 / 0.7270 / -0.1254 | 0.1379 / 0.1379 / +0.0000 | 0.1615 / 0.1925 / -0.0310 | 0.0000 / 0.0000 / +0.0000 | auc -0.1254 |
| degree | random_persistent | 0.5878 / 0.6819 / -0.0941 | 0.1013 / 0.1017 / -0.0004 | 0.1270 / 0.1426 / -0.0156 | 0.2323 / 0.2533 / -0.0210 | auc -0.0941 |
| degree | region_persistent | 0.5742 / 0.6684 / -0.0942 | 0.0917 / 0.0909 / +0.0008 | 0.1118 / 0.1199 / -0.0081 | 0.3462 / 0.3642 / -0.0180 | auc -0.0942 |
| degree | scattered | 0.5999 / 0.7229 / -0.1230 | 0.1333 / 0.1329 / +0.0004 | 0.1560 / 0.1888 / -0.0328 | 0.0282 / 0.0270 / +0.0012 | auc -0.1230 |
| fitted_mechanistic | bridge_persistent | 0.7256 / 0.7886 / -0.0630 | 0.1714 / 0.1767 / -0.0053 | 0.2058 / 0.2224 / -0.0166 | 0.1508 / 0.1307 / +0.0201 | auc -0.0630 |
| fitted_mechanistic | degree_matched | 0.7352 / 0.7992 / -0.0640 | 0.1711 / 0.1755 / -0.0044 | 0.1856 / 0.2236 / -0.0380 | 0.1484 / 0.1220 / +0.0264 | auc -0.0640 |
| fitted_mechanistic | full | 0.8608 / 0.9146 / -0.0538 | 0.3135 / 0.3135 / +0.0000 | 0.3295 / 0.3674 / -0.0379 | 0.0000 / 0.0000 / +0.0000 | auc -0.0538 |
| fitted_mechanistic | random_persistent | 0.8130 / 0.8525 / -0.0395 | 0.2352 / 0.2383 / -0.0031 | 0.2798 / 0.2933 / -0.0135 | 0.0279 / 0.0303 / -0.0024 | auc -0.0395 |
| fitted_mechanistic | region_persistent | 0.7709 / 0.8256 / -0.0547 | 0.2038 / 0.2094 / -0.0056 | 0.2358 / 0.2881 / -0.0523 | 0.0730 / 0.0463 / +0.0267 | auc -0.0547 |
| fitted_mechanistic | scattered | 0.8488 / 0.9058 / -0.0570 | 0.2940 / 0.2924 / +0.0016 | 0.3242 / 0.3667 / -0.0425 | 0.0370 / 0.0334 / +0.0036 | auc -0.0570 |
| logistic | bridge_persistent | 0.5606 / 0.7197 / -0.1591 | 0.1117 / 0.1280 / -0.0163 | 0.1649 / 0.1989 / -0.0340 | 0.3837 / 0.3032 / +0.0805 | auc -0.1591 |
| logistic | degree_matched | 0.5672 / 0.7388 / -0.1716 | 0.1068 / 0.1220 / -0.0152 | 0.1504 / 0.2021 / -0.0517 | 0.3918 / 0.2987 / +0.0931 | auc -0.1716 |
| logistic | full | 0.8282 / 0.9030 / -0.0748 | 0.2975 / 0.2889 / +0.0086 | 0.3250 / 0.3737 / -0.0487 | 0.0000 / 0.0000 / +0.0000 | auc -0.0748 |
| logistic | random_persistent | 0.7263 / 0.8062 / -0.0799 | 0.1765 / 0.1833 / -0.0068 | 0.2337 / 0.2614 / -0.0277 | 0.2542 / 0.2118 / +0.0424 | auc -0.0799 |
| logistic | region_persistent | 0.6106 / 0.7654 / -0.1548 | 0.1497 / 0.1599 / -0.0102 | 0.1952 / 0.2546 / -0.0594 | 0.2127 / 0.2166 / -0.0039 | auc -0.1548 |
| logistic | scattered | 0.7753 / 0.8775 / -0.1022 | 0.2509 / 0.2413 / +0.0096 | 0.2954 / 0.3204 / -0.0250 | 0.0418 / 0.0343 / +0.0075 | auc -0.1022 |
| mask_only | bridge_persistent | 0.5146 / 0.6199 / -0.1053 | 0.0836 / 0.0865 / -0.0029 | 0.0725 / 0.0611 / +0.0114 | 0.1016 / 0.2169 / -0.1153 | auc -0.1053 |
| mask_only | degree_matched | 0.5085 / 0.6151 / -0.1066 | 0.0844 / 0.0869 / -0.0025 | 0.0700 / 0.0687 / +0.0013 | 0.1055 / 0.2112 / -0.1057 | auc -0.1066 |
| mask_only | full | 0.5000 / 0.6543 / -0.1543 | 0.0973 / 0.0973 / -0.0000 | 0.0743 / 0.1484 / -0.0741 | 0.0000 / 0.0000 / +0.0000 | auc -0.1543 |
| mask_only | random_persistent | 0.4948 / 0.5971 / -0.1023 | 0.0822 / 0.0748 / +0.0074 | 0.0642 / 0.0375 / +0.0267 | 0.0793 / 0.1608 / -0.0815 | auc -0.1023 |
| mask_only | region_persistent | 0.5274 / 0.6205 / -0.0931 | 0.0850 / 0.0819 / +0.0031 | 0.0810 / 0.0703 / +0.0107 | 0.0968 / 0.2767 / -0.1799 | auc -0.0931 |
| mask_only | scattered | 0.5014 / 0.6458 / -0.1444 | 0.0960 / 0.0955 / +0.0005 | 0.0780 / 0.0706 / +0.0074 | 0.0592 / 0.0325 / +0.0267 | auc -0.1444 |
| observed_neighbor | bridge_persistent | 0.5613 / 0.5815 / -0.0202 | 0.0942 / 0.0791 / +0.0151 | 0.1528 / 0.1690 / -0.0162 | 0.4534 / 0.4787 / -0.0253 |  |
| observed_neighbor | degree_matched | 0.5573 / 0.6007 / -0.0434 | 0.0912 / 0.0845 / +0.0067 | 0.1393 / 0.1864 / -0.0471 | 0.4546 / 0.4799 / -0.0253 | auc -0.0434 |
| observed_neighbor | full | 0.8235 / 0.8872 / -0.0637 | 0.2630 / 0.2741 / -0.0111 | 0.3270 / 0.3789 / -0.0519 | 0.0000 / 0.0000 / +0.0000 | auc -0.0637 |
| observed_neighbor | random_persistent | 0.6947 / 0.7214 / -0.0267 | 0.1398 / 0.1293 / +0.0105 | 0.2184 / 0.2388 / -0.0204 | 0.3555 / 0.3675 / -0.0120 |  |
| observed_neighbor | region_persistent | 0.6078 / 0.6574 / -0.0496 | 0.1156 / 0.1144 / +0.0012 | 0.1764 / 0.2270 / -0.0506 | 0.2909 / 0.3233 / -0.0324 | auc -0.0496 |
| observed_neighbor | scattered | 0.7197 / 0.7540 / -0.0343 | 0.2009 / 0.1859 / +0.0150 | 0.2851 / 0.3173 / -0.0322 | 0.0451 / 0.0409 / +0.0042 | auc -0.0343 |
| prevalence | bridge_persistent | 0.5000 / 0.6010 / -0.1010 | 0.0762 / 0.0758 / +0.0004 | 0.0607 / 0.0883 / -0.0276 | 0.1457 / 0.3131 / -0.1674 | auc -0.1010 |
| prevalence | degree_matched | 0.5000 / 0.6028 / -0.1028 | 0.0760 / 0.0763 / -0.0003 | 0.0613 / 0.0856 / -0.0243 | 0.1448 / 0.3263 / -0.1815 | auc -0.1028 |
| prevalence | full | 0.5000 / 0.6543 / -0.1543 | 0.0973 / 0.0973 / -0.0000 | 0.0743 / 0.1484 / -0.0741 | 0.0000 / 0.0000 / +0.0000 | auc -0.1543 |
| prevalence | random_persistent | 0.5000 / 0.6104 / -0.1104 | 0.0775 / 0.0781 / -0.0006 | 0.0613 / 0.1038 / -0.0425 | 0.1265 / 0.2272 / -0.1007 | auc -0.1104 |
| prevalence | region_persistent | 0.5000 / 0.6049 / -0.1049 | 0.0775 / 0.0767 / +0.0008 | 0.0600 / 0.0904 / -0.0304 | 0.1391 / 0.3194 / -0.1803 | auc -0.1049 |
| prevalence | scattered | 0.5000 / 0.6503 / -0.1503 | 0.0951 / 0.0953 / -0.0002 | 0.0729 / 0.1463 / -0.0734 | 0.0150 / 0.0228 / -0.0078 | auc -0.1503 |

Flagged cells (|ΔAP|>0.02 or |ΔAUC|>0.03): 40
- boosted_trees / bridge_persistent: auc -0.0928
- boosted_trees / degree_matched: auc -0.0984
- boosted_trees / full: auc -0.0635
- boosted_trees / random_persistent: auc -0.0588
- boosted_trees / region_persistent: auc -0.0723
- boosted_trees / scattered: auc -0.0598
- degree / bridge_persistent: auc -0.1006
- degree / degree_matched: auc -0.1021
- degree / full: auc -0.1254
- degree / random_persistent: auc -0.0941
- degree / region_persistent: auc -0.0942
- degree / scattered: auc -0.1230
- fitted_mechanistic / bridge_persistent: auc -0.0630
- fitted_mechanistic / degree_matched: auc -0.0640
- fitted_mechanistic / full: auc -0.0538
- fitted_mechanistic / random_persistent: auc -0.0395
- fitted_mechanistic / region_persistent: auc -0.0547
- fitted_mechanistic / scattered: auc -0.0570
- logistic / bridge_persistent: auc -0.1591
- logistic / degree_matched: auc -0.1716
- logistic / full: auc -0.0748
- logistic / random_persistent: auc -0.0799
- logistic / region_persistent: auc -0.1548
- logistic / scattered: auc -0.1022
- mask_only / bridge_persistent: auc -0.1053
- mask_only / degree_matched: auc -0.1066
- mask_only / full: auc -0.1543
- mask_only / random_persistent: auc -0.1023
- mask_only / region_persistent: auc -0.0931
- mask_only / scattered: auc -0.1444
- observed_neighbor / degree_matched: auc -0.0434
- observed_neighbor / full: auc -0.0637
- observed_neighbor / region_persistent: auc -0.0496
- observed_neighbor / scattered: auc -0.0343
- prevalence / bridge_persistent: auc -0.1010
- prevalence / degree_matched: auc -0.1028
- prevalence / full: auc -0.1543
- prevalence / random_persistent: auc -0.1104
- prevalence / region_persistent: auc -0.1049
- prevalence / scattered: auc -0.1503

Required rows present: fitted_mechanistic/full=yes, fitted_mechanistic/bridge_persistent=yes, boosted_trees/full=yes, boosted_trees/bridge_persistent=yes

### C05 notes

Run on Windows with `stage2.py --data results/stage1/run_a --out results/stage2/run_a`, then `verify_classical.py` and `tools/compare_september.py`. No code was changed to close a gap.

September's log is the all-host population. On candidates, prevalence AUC is 0.5 by construction, while September reports 0.60 to 0.65. So the all-host table above is the comparison, and the candidate table is for reference only. Every all-host value matches the earlier Linux Step A (Stitchnotes/STITCH_Step_A_Sanity_Check.md) to 4 decimals.

### C05 flag causes (all-host, 8 flags)

| Flag | Cause in code |
|---|---|
| observed_neighbor, all 5 masked conditions, AUC +0.034 to +0.076 | Feature definition and candidate rule. `classical.columns` uses column 17 only, the count of currently visible I neighbours, passed through a logistic fit (P = 0.045 at zero count). The E/I/P gate in `classical.predict` sets known hosts to 0, so unknown positives with zero exposure outrank known negatives instead of tying with them. The measured tie effect (0.043 to 0.079) is the same size as each gap. Bridge and region host sets also differ from September (88 to 93 and 30 to 82 of 102 hosts shared). |
| logistic, bridge / degree_matched / region, AUC -0.070 to -0.085 | Feature definition. `classical.design` builds 22 inputs (8 current, 8 window means, 6 neighbour sums) from 294,912 rows. September's scaler had 32 inputs and 423,936 rows, so the inputs differ before any masking. Persistent structural masks change the neighbour-sum features most. |

Not flagged but systematic: recall_at_budget is lower in every row (up to -0.074). The aggregation order differs: the rebuild pools hits over primary origins within each outbreak, while September used a per-origin denominator (episode 3000_0 check: 0.2789 vs 0.3327). The late share for prevalence and mask_only is lower because of the tie rule: shared seeded permutations here, September's rule unknown.

Accept: every flag has a code cause; no code changed to close a gap; fitted_mechanistic and boosted_trees rows are present for full and bridge_persistent.

## Spec v1.0.1 clarifications

Decided after C05 was accepted and before any neural training.

1. **Training origins.** The arithmetic is confirmed. There are 24 training episodes × 256 hosts = 6,144 host-episodes. September: 423,936 = 6,144 × 69, where 69 = 23 origins (t = 7, 9, …, 51) × 3 masks. Ours at C05: 294,912 = 6,144 × 48, where 48 = 16 origins (13..43) × 3 masks. From now on, all classical models (and, later, all neural models) train on `pilot.TRAIN_ORIGINS` = 7..51 step 2 with the 3 training masks. The fitted_mechanistic prior is taken at the same window starts. Validation selection and scoring stay on the primary origins 13..43.
2. **Features.** The 22 logistic and boosted-tree features are documented in `docs/methods/classical_features.md`. September's 32 features are not reconstructed.
3. **Known hosts.** Observer-known hosts score `pilot.KNOWN_SCORE` = -1 for every method, so they rank strictly below every candidate and nothing ties with them. Brier, ECE and reliability clip scores to [0, 1]. Candidate-only metrics remain primary.
4. **recall_at_budget.** PRIMARY is `recall_at_budget`, pooled within each outbreak: Σ hits / Σ positives over the primary origins. SECONDARY is `recall_at_budget_per_origin`, the mean over primary origins of hits / positives. Origins with zero positives are left out, and their count is recorded as `origins_zero_positive` in episode_metrics.csv.
## C05 Step A (C05b rerun)

### Verification
- Stage 1 T01: PASS
- Stage 1 T02: PASS
- Stage 1 T03: PASS
- Stage 1 T04: PASS
- Stage 1 T05: PASS
- Stage 1 T06: PASS
- Stage 1 T07: PASS
- Stage 1 T08: PASS
- Stage 1 T09: PASS
- Stage 1 T10: PASS
- Stage 1 T10b: PASS
- Stage 1 T10c: PASS
- Stage 1 T10d: PASS
- Stage 2 test_01_future_event_and_earliest_positive_lead: PASS
- Stage 2 test_02_candidate_filter_ties_and_budget: PASS
- Stage 2 test_03_mask_only_and_causal_model_inputs: PASS
- Stage 2 test_04_filter_mass_and_observed_state: PASS
- Stage 2 test_05_frozen_splits_and_complete_score_grid: PASS
- Stage 2 test_06_all_saved_scores_respect_confirmation: PASS
- Stage 2 test_07_bridge_diagnostic_and_primary_population: PASS

### bridge_persistent candidate prevalence (September 0.0758)

| rule | graph_macro | pooled | positives | eligible |
|---|---|---|---|---|
| E/I | 0.0758 | 0.0690 | 3674 | 53284 |
| E/I/P | 0.0762 | 0.0692 | 3674 | 53112 |

### Population: all

| model | condition | auc Sep / old / new / Δ(new-Sep) | ap Sep / old / new / Δ(new-Sep) | recall_at_budget Sep / old / new / Δ(new-Sep) | late_detection_share Sep / old / new / Δ(new-Sep) | recall per-origin (new) | flag |
|---|---|---|---|---|---|---|---|
| boosted_trees | bridge_persistent | 0.7460 / 0.7284 / 0.7242 / -0.0218 | 0.1477 / 0.1489 / 0.1474 / -0.0003 | 0.2273 / 0.1872 / 0.1880 / -0.0393 | 0.2076 / 0.2275 / 0.2257 / +0.0181 | 0.2058 |  |
| boosted_trees | degree_matched | 0.7544 / 0.7296 / 0.7295 / -0.0249 | 0.1489 / 0.1446 / 0.1458 / -0.0031 | 0.2244 / 0.1877 / 0.1892 / -0.0352 | 0.2076 / 0.2175 / 0.2184 / +0.0108 | 0.2106 |  |
| boosted_trees | full | 0.9063 / 0.9027 / 0.9029 / -0.0034 | 0.3035 / 0.3024 / 0.3034 / -0.0001 | 0.3820 / 0.3253 / 0.3319 / -0.0501 | 0.0000 / 0.0000 / 0.0000 / +0.0000 | 0.3672 |  |
| boosted_trees | random_persistent | 0.8328 / 0.8295 / 0.8266 / -0.0062 | 0.2247 / 0.2237 / 0.2250 / +0.0003 | 0.3031 / 0.2757 / 0.2744 / -0.0287 | 0.0394 / 0.0376 / 0.0376 / -0.0018 | 0.3022 |  |
| boosted_trees | region_persistent | 0.7927 / 0.7830 / 0.7813 / -0.0114 | 0.1941 / 0.1936 / 0.1942 / +0.0001 | 0.2977 / 0.2462 / 0.2366 / -0.0611 | 0.0559 / 0.0763 / 0.0808 / +0.0249 | 0.2696 |  |
| boosted_trees | scattered | 0.8809 / 0.8874 / 0.8861 / +0.0052 | 0.2570 / 0.2729 / 0.2681 / +0.0111 | 0.3278 / 0.3092 / 0.3101 / -0.0177 | 0.0352 / 0.0406 / 0.0433 / +0.0081 | 0.3445 |  |
| degree | bridge_persistent | 0.6485 / 0.6484 / 0.6484 / -0.0001 | 0.0837 / 0.0837 / 0.0837 / -0.0000 | 0.1120 / 0.1024 / 0.1024 / -0.0096 | 0.3975 / 0.4020 / 0.4020 / +0.0045 | 0.1103 |  |
| degree | degree_matched | 0.6503 / 0.6481 / 0.6481 / -0.0022 | 0.0840 / 0.0834 / 0.0834 / -0.0006 | 0.1119 / 0.1024 / 0.1024 / -0.0095 | 0.4023 / 0.4023 / 0.4023 / +0.0000 | 0.1103 |  |
| degree | full | 0.7270 / 0.7270 / 0.7270 / +0.0000 | 0.1379 / 0.1379 / 0.1379 / +0.0000 | 0.1925 / 0.1615 / 0.1615 / -0.0310 | 0.0000 / 0.0000 / 0.0000 / +0.0000 | 0.1845 |  |
| degree | random_persistent | 0.6819 / 0.6816 / 0.6816 / -0.0003 | 0.1017 / 0.1013 / 0.1013 / -0.0004 | 0.1426 / 0.1270 / 0.1270 / -0.0156 | 0.2533 / 0.2323 / 0.2323 / -0.0210 | 0.1419 |  |
| degree | region_persistent | 0.6684 / 0.6681 / 0.6681 / -0.0003 | 0.0909 / 0.0917 / 0.0917 / +0.0008 | 0.1199 / 0.1118 / 0.1118 / -0.0081 | 0.3642 / 0.3462 / 0.3462 / -0.0180 | 0.1249 |  |
| degree | scattered | 0.7229 / 0.7227 / 0.7227 / -0.0002 | 0.1329 / 0.1333 / 0.1333 / +0.0004 | 0.1888 / 0.1560 / 0.1560 / -0.0328 | 0.0270 / 0.0282 / 0.0282 / +0.0012 | 0.1795 |  |
| fitted_mechanistic | bridge_persistent | 0.7886 / 0.7895 / 0.7903 / +0.0017 | 0.1767 / 0.1714 / 0.1718 / -0.0049 | 0.2224 / 0.2058 / 0.2072 / -0.0152 | 0.1307 / 0.1508 / 0.1544 / +0.0237 | 0.2199 |  |
| fitted_mechanistic | degree_matched | 0.7992 / 0.7964 / 0.7968 / -0.0024 | 0.1755 / 0.1711 / 0.1719 / -0.0036 | 0.2236 / 0.1856 / 0.1883 / -0.0353 | 0.1220 / 0.1484 / 0.1523 / +0.0303 | 0.2121 |  |
| fitted_mechanistic | full | 0.9146 / 0.9146 / 0.9146 / +0.0000 | 0.3135 / 0.3135 / 0.3135 / +0.0000 | 0.3674 / 0.3295 / 0.3295 / -0.0379 | 0.0000 / 0.0000 / 0.0000 / +0.0000 | 0.3682 |  |
| fitted_mechanistic | random_persistent | 0.8525 / 0.8604 / 0.8607 / +0.0082 | 0.2383 / 0.2352 / 0.2358 / -0.0025 | 0.2933 / 0.2798 / 0.2812 / -0.0121 | 0.0303 / 0.0279 / 0.0270 / -0.0033 | 0.3026 |  |
| fitted_mechanistic | region_persistent | 0.8256 / 0.8236 / 0.8245 / -0.0011 | 0.2094 / 0.2038 / 0.2050 / -0.0044 | 0.2881 / 0.2358 / 0.2346 / -0.0535 | 0.0463 / 0.0730 / 0.0700 / +0.0237 | 0.2640 |  |
| fitted_mechanistic | scattered | 0.9058 / 0.9057 / 0.9057 / -0.0001 | 0.2924 / 0.2940 / 0.2940 / +0.0016 | 0.3667 / 0.3242 / 0.3240 / -0.0427 | 0.0334 / 0.0370 / 0.0370 / +0.0036 | 0.3638 |  |
| logistic | bridge_persistent | 0.7197 / 0.6496 / 0.6464 / -0.0733 | 0.1280 / 0.1117 / 0.1108 / -0.0172 | 0.1989 / 0.1649 / 0.1639 / -0.0350 | 0.3032 / 0.3837 / 0.4069 / +0.1037 | 0.1831 | auc -0.0733 |
| logistic | degree_matched | 0.7388 / 0.6536 / 0.6508 / -0.0880 | 0.1220 / 0.1068 / 0.1054 / -0.0166 | 0.2021 / 0.1504 / 0.1553 / -0.0468 | 0.2987 / 0.3918 / 0.4099 / +0.1112 | 0.1700 | auc -0.0880 |
| logistic | full | 0.9030 / 0.8913 / 0.8934 / -0.0096 | 0.2889 / 0.2975 / 0.3010 / +0.0121 | 0.3737 / 0.3250 / 0.3231 / -0.0506 | 0.0000 / 0.0000 / 0.0000 / +0.0000 | 0.3624 |  |
| logistic | random_persistent | 0.8062 / 0.7909 / 0.7910 / -0.0152 | 0.1833 / 0.1765 / 0.1734 / -0.0099 | 0.2614 / 0.2337 / 0.2396 / -0.0218 | 0.2118 / 0.2542 / 0.2767 / +0.0649 | 0.2610 |  |
| logistic | region_persistent | 0.7654 / 0.6850 / 0.6830 / -0.0824 | 0.1599 / 0.1497 / 0.1490 / -0.0109 | 0.2546 / 0.1952 / 0.1972 / -0.0574 | 0.2166 / 0.2127 / 0.2308 / +0.0142 | 0.2249 | auc -0.0824 |
| logistic | scattered | 0.8775 / 0.8516 / 0.8563 / -0.0212 | 0.2413 / 0.2509 / 0.2523 / +0.0110 | 0.3204 / 0.2954 / 0.2961 / -0.0243 | 0.0343 / 0.0418 / 0.0460 / +0.0117 | 0.3254 |  |
| mask_only | bridge_persistent | 0.6199 / 0.6088 / 0.6171 / -0.0028 | 0.0865 / 0.0836 / 0.0823 / -0.0042 | 0.0611 / 0.0725 / 0.0705 / +0.0094 | 0.2169 / 0.1016 / 0.1719 / -0.0450 | 0.0750 |  |
| mask_only | degree_matched | 0.6151 / 0.6024 / 0.6112 / -0.0039 | 0.0869 / 0.0844 / 0.0829 / -0.0040 | 0.0687 / 0.0700 / 0.0672 / -0.0015 | 0.2112 / 0.1055 / 0.1767 / -0.0345 | 0.0736 |  |
| mask_only | full | 0.6543 / 0.6543 / 0.6543 / +0.0000 | 0.0973 / 0.0973 / 0.0973 / -0.0000 | 0.1484 / 0.0743 / 0.0743 / -0.0741 | 0.0000 / 0.0000 / 0.0000 / +0.0000 | 0.0873 |  |
| mask_only | random_persistent | 0.5971 / 0.6004 / 0.5987 / +0.0016 | 0.0748 / 0.0822 / 0.0794 / +0.0046 | 0.0375 / 0.0642 / 0.0604 / +0.0229 | 0.1608 / 0.0793 / 0.1343 / -0.0265 | 0.0700 |  |
| mask_only | region_persistent | 0.6205 / 0.6238 / 0.6272 / +0.0067 | 0.0819 / 0.0850 / 0.0829 / +0.0010 | 0.0703 / 0.0810 / 0.0746 / +0.0043 | 0.2767 / 0.0968 / 0.1611 / -0.1156 | 0.0859 |  |
| mask_only | scattered | 0.6458 / 0.6502 / 0.6558 / +0.0100 | 0.0955 / 0.0960 / 0.0944 / -0.0011 | 0.0706 / 0.0780 / 0.0730 / +0.0024 | 0.0325 / 0.0592 / 0.0547 / +0.0222 | 0.0874 |  |
| observed_neighbor | bridge_persistent | 0.5815 / 0.6577 / 0.6577 / +0.0762 | 0.0791 / 0.0942 / 0.0942 / +0.0151 | 0.1690 / 0.1528 / 0.1528 / -0.0162 | 0.4787 / 0.4534 / 0.4534 / -0.0253 | 0.1725 | auc +0.0762 |
| observed_neighbor | degree_matched | 0.6007 / 0.6534 / 0.6534 / +0.0527 | 0.0845 / 0.0912 / 0.0912 / +0.0067 | 0.1864 / 0.1393 / 0.1393 / -0.0471 | 0.4799 / 0.4546 / 0.4546 / -0.0253 | 0.1528 | auc +0.0527 |
| observed_neighbor | full | 0.8872 / 0.8885 / 0.8885 / +0.0013 | 0.2741 / 0.2630 / 0.2630 / -0.0111 | 0.3789 / 0.3270 / 0.3270 / -0.0519 | 0.0000 / 0.0000 / 0.0000 / +0.0000 | 0.3686 |  |
| observed_neighbor | random_persistent | 0.7214 / 0.7696 / 0.7696 / +0.0482 | 0.1293 / 0.1398 / 0.1398 / +0.0105 | 0.2388 / 0.2184 / 0.2184 / -0.0204 | 0.3675 / 0.3555 / 0.3555 / -0.0120 | 0.2398 | auc +0.0482 |
| observed_neighbor | region_persistent | 0.6574 / 0.6910 / 0.6910 / +0.0336 | 0.1144 / 0.1156 / 0.1156 / +0.0012 | 0.2270 / 0.1764 / 0.1764 / -0.0506 | 0.3233 / 0.2909 / 0.2909 / -0.0324 | 0.2021 | auc +0.0336 |
| observed_neighbor | scattered | 0.7540 / 0.8114 / 0.8114 / +0.0574 | 0.1859 / 0.2009 / 0.2009 / +0.0150 | 0.3173 / 0.2851 / 0.2851 / -0.0322 | 0.0409 / 0.0451 / 0.0451 / +0.0042 | 0.3203 | auc +0.0574 |
| prevalence | bridge_persistent | 0.6010 / 0.6023 / 0.6023 / +0.0013 | 0.0758 / 0.0762 / 0.0762 / +0.0004 | 0.0883 / 0.0607 / 0.0607 / -0.0276 | 0.3131 / 0.1457 / 0.1457 / -0.1674 | 0.0680 |  |
| prevalence | degree_matched | 0.6028 / 0.6018 / 0.6018 / -0.0010 | 0.0763 / 0.0760 / 0.0760 / -0.0003 | 0.0856 / 0.0613 / 0.0613 / -0.0243 | 0.3263 / 0.1448 / 0.1448 / -0.1815 | 0.0690 |  |
| prevalence | full | 0.6543 / 0.6543 / 0.6543 / +0.0000 | 0.0973 / 0.0973 / 0.0973 / -0.0000 | 0.1484 / 0.0743 / 0.0743 / -0.0741 | 0.0000 / 0.0000 / 0.0000 / +0.0000 | 0.0873 |  |
| prevalence | random_persistent | 0.6104 / 0.6101 / 0.6101 / -0.0003 | 0.0781 / 0.0775 / 0.0775 / -0.0006 | 0.1038 / 0.0613 / 0.0613 / -0.0425 | 0.2272 / 0.1265 / 0.1265 / -0.1007 | 0.0677 |  |
| prevalence | region_persistent | 0.6049 / 0.6046 / 0.6046 / -0.0003 | 0.0767 / 0.0775 / 0.0775 / +0.0008 | 0.0904 / 0.0600 / 0.0600 / -0.0304 | 0.3194 / 0.1391 / 0.1391 / -0.1803 | 0.0679 |  |
| prevalence | scattered | 0.6503 / 0.6500 / 0.6500 / -0.0003 | 0.0953 / 0.0951 / 0.0951 / -0.0002 | 0.1463 / 0.0729 / 0.0729 / -0.0734 | 0.0228 / 0.0150 / 0.0150 / -0.0078 | 0.0861 |  |

Flagged cells (|ΔAP|>0.02 or |ΔAUC|>0.03 vs September): 8
- logistic / bridge_persistent: auc -0.0733
- logistic / degree_matched: auc -0.0880
- logistic / region_persistent: auc -0.0824
- observed_neighbor / bridge_persistent: auc +0.0762
- observed_neighbor / degree_matched: auc +0.0527
- observed_neighbor / random_persistent: auc +0.0482
- observed_neighbor / region_persistent: auc +0.0336
- observed_neighbor / scattered: auc +0.0574

### Population: candidate

| model | condition | auc Sep / old / new / Δ(new-Sep) | ap Sep / old / new / Δ(new-Sep) | recall_at_budget Sep / old / new / Δ(new-Sep) | late_detection_share Sep / old / new / Δ(new-Sep) | recall per-origin (new) | flag |
|---|---|---|---|---|---|---|---|
| boosted_trees | bridge_persistent | 0.7460 / 0.6532 / 0.6483 / -0.0977 | 0.1477 / 0.1489 / 0.1474 / -0.0003 | 0.2273 / 0.1872 / 0.1880 / -0.0393 | 0.2076 / 0.2275 / 0.2257 / +0.0181 | 0.2058 |  |
| boosted_trees | degree_matched | 0.7544 / 0.6560 / 0.6559 / -0.0985 | 0.1489 / 0.1446 / 0.1458 / -0.0031 | 0.2244 / 0.1877 / 0.1892 / -0.0352 | 0.2076 / 0.2175 / 0.2184 / +0.0108 | 0.2106 |  |
| boosted_trees | full | 0.9063 / 0.8428 / 0.8432 / -0.0631 | 0.3035 / 0.3024 / 0.3034 / -0.0001 | 0.3820 / 0.3253 / 0.3319 / -0.0501 | 0.0000 / 0.0000 / 0.0000 / +0.0000 | 0.3672 |  |
| boosted_trees | random_persistent | 0.8328 / 0.7740 / 0.7705 / -0.0623 | 0.2247 / 0.2237 / 0.2250 / +0.0003 | 0.3031 / 0.2757 / 0.2744 / -0.0287 | 0.0394 / 0.0376 / 0.0376 / -0.0018 | 0.3022 |  |
| boosted_trees | region_persistent | 0.7927 / 0.7204 / 0.7183 / -0.0744 | 0.1941 / 0.1936 / 0.1942 / +0.0001 | 0.2977 / 0.2462 / 0.2366 / -0.0611 | 0.0559 / 0.0763 / 0.0808 / +0.0249 | 0.2696 |  |
| boosted_trees | scattered | 0.8809 / 0.8211 / 0.8196 / -0.0613 | 0.2570 / 0.2729 / 0.2681 / +0.0111 | 0.3278 / 0.3092 / 0.3101 / -0.0177 | 0.0352 / 0.0406 / 0.0433 / +0.0081 | 0.3445 |  |
| degree | bridge_persistent | 0.6485 / 0.5479 / 0.5479 / -0.1006 | 0.0837 / 0.0837 / 0.0837 / -0.0000 | 0.1120 / 0.1024 / 0.1024 / -0.0096 | 0.3975 / 0.4020 / 0.4020 / +0.0045 | 0.1103 |  |
| degree | degree_matched | 0.6503 / 0.5482 / 0.5482 / -0.1021 | 0.0840 / 0.0834 / 0.0834 / -0.0006 | 0.1119 / 0.1024 / 0.1024 / -0.0095 | 0.4023 / 0.4023 / 0.4023 / +0.0000 | 0.1103 |  |
| degree | full | 0.7270 / 0.6016 / 0.6016 / -0.1254 | 0.1379 / 0.1379 / 0.1379 / +0.0000 | 0.1925 / 0.1615 / 0.1615 / -0.0310 | 0.0000 / 0.0000 / 0.0000 / +0.0000 | 0.1845 |  |
| degree | random_persistent | 0.6819 / 0.5878 / 0.5878 / -0.0941 | 0.1017 / 0.1013 / 0.1013 / -0.0004 | 0.1426 / 0.1270 / 0.1270 / -0.0156 | 0.2533 / 0.2323 / 0.2323 / -0.0210 | 0.1419 |  |
| degree | region_persistent | 0.6684 / 0.5742 / 0.5742 / -0.0942 | 0.0909 / 0.0917 / 0.0917 / +0.0008 | 0.1199 / 0.1118 / 0.1118 / -0.0081 | 0.3642 / 0.3462 / 0.3462 / -0.0180 | 0.1249 |  |
| degree | scattered | 0.7229 / 0.5999 / 0.5999 / -0.1230 | 0.1329 / 0.1333 / 0.1333 / +0.0004 | 0.1888 / 0.1560 / 0.1560 / -0.0328 | 0.0270 / 0.0282 / 0.0282 / +0.0012 | 0.1795 |  |
| fitted_mechanistic | bridge_persistent | 0.7886 / 0.7256 / 0.7266 / -0.0620 | 0.1767 / 0.1714 / 0.1718 / -0.0049 | 0.2224 / 0.2058 / 0.2072 / -0.0152 | 0.1307 / 0.1508 / 0.1544 / +0.0237 | 0.2199 |  |
| fitted_mechanistic | degree_matched | 0.7992 / 0.7352 / 0.7357 / -0.0635 | 0.1755 / 0.1711 / 0.1719 / -0.0036 | 0.2236 / 0.1856 / 0.1883 / -0.0353 | 0.1220 / 0.1484 / 0.1523 / +0.0303 | 0.2121 |  |
| fitted_mechanistic | full | 0.9146 / 0.8608 / 0.8608 / -0.0538 | 0.3135 / 0.3135 / 0.3135 / +0.0000 | 0.3674 / 0.3295 / 0.3295 / -0.0379 | 0.0000 / 0.0000 / 0.0000 / +0.0000 | 0.3682 |  |
| fitted_mechanistic | random_persistent | 0.8525 / 0.8130 / 0.8135 / -0.0390 | 0.2383 / 0.2352 / 0.2358 / -0.0025 | 0.2933 / 0.2798 / 0.2812 / -0.0121 | 0.0303 / 0.0279 / 0.0270 / -0.0033 | 0.3026 |  |
| fitted_mechanistic | region_persistent | 0.8256 / 0.7709 / 0.7721 / -0.0535 | 0.2094 / 0.2038 / 0.2050 / -0.0044 | 0.2881 / 0.2358 / 0.2346 / -0.0535 | 0.0463 / 0.0730 / 0.0700 / +0.0237 | 0.2640 |  |
| fitted_mechanistic | scattered | 0.9058 / 0.8488 / 0.8488 / -0.0570 | 0.2924 / 0.2940 / 0.2940 / +0.0016 | 0.3667 / 0.3242 / 0.3240 / -0.0427 | 0.0334 / 0.0370 / 0.0370 / +0.0036 | 0.3638 |  |
| logistic | bridge_persistent | 0.7197 / 0.5606 / 0.5567 / -0.1630 | 0.1280 / 0.1117 / 0.1108 / -0.0172 | 0.1989 / 0.1649 / 0.1639 / -0.0350 | 0.3032 / 0.3837 / 0.4069 / +0.1037 | 0.1831 |  |
| logistic | degree_matched | 0.7388 / 0.5672 / 0.5635 / -0.1753 | 0.1220 / 0.1068 / 0.1054 / -0.0166 | 0.2021 / 0.1504 / 0.1553 / -0.0468 | 0.2987 / 0.3918 / 0.4099 / +0.1112 | 0.1700 |  |
| logistic | full | 0.9030 / 0.8282 / 0.8307 / -0.0723 | 0.2889 / 0.2975 / 0.3010 / +0.0121 | 0.3737 / 0.3250 / 0.3231 / -0.0506 | 0.0000 / 0.0000 / 0.0000 / +0.0000 | 0.3624 |  |
| logistic | random_persistent | 0.8062 / 0.7263 / 0.7253 / -0.0809 | 0.1833 / 0.1765 / 0.1734 / -0.0099 | 0.2614 / 0.2337 / 0.2396 / -0.0218 | 0.2118 / 0.2542 / 0.2767 / +0.0649 | 0.2610 |  |
| logistic | region_persistent | 0.7654 / 0.6106 / 0.6082 / -0.1572 | 0.1599 / 0.1497 / 0.1490 / -0.0109 | 0.2546 / 0.1952 / 0.1972 / -0.0574 | 0.2166 / 0.2127 / 0.2308 / +0.0142 | 0.2249 |  |
| logistic | scattered | 0.8775 / 0.7753 / 0.7808 / -0.0967 | 0.2413 / 0.2509 / 0.2523 / +0.0110 | 0.3204 / 0.2954 / 0.2961 / -0.0243 | 0.0343 / 0.0418 / 0.0460 / +0.0117 | 0.3254 |  |
| mask_only | bridge_persistent | 0.6199 / 0.5146 / 0.5195 / -0.1004 | 0.0865 / 0.0836 / 0.0823 / -0.0042 | 0.0611 / 0.0725 / 0.0705 / +0.0094 | 0.2169 / 0.1016 / 0.1719 / -0.0450 | 0.0750 |  |
| mask_only | degree_matched | 0.6151 / 0.5085 / 0.5140 / -0.1011 | 0.0869 / 0.0844 / 0.0829 / -0.0040 | 0.0687 / 0.0700 / 0.0672 / -0.0015 | 0.2112 / 0.1055 / 0.1767 / -0.0345 | 0.0736 |  |
| mask_only | full | 0.6543 / 0.5000 / 0.5000 / -0.1543 | 0.0973 / 0.0973 / 0.0973 / -0.0000 | 0.1484 / 0.0743 / 0.0743 / -0.0741 | 0.0000 / 0.0000 / 0.0000 / +0.0000 | 0.0873 |  |
| mask_only | random_persistent | 0.5971 / 0.4948 / 0.4885 / -0.1086 | 0.0748 / 0.0822 / 0.0794 / +0.0046 | 0.0375 / 0.0642 / 0.0604 / +0.0229 | 0.1608 / 0.0793 / 0.1343 / -0.0265 | 0.0700 |  |
| mask_only | region_persistent | 0.6205 / 0.5274 / 0.5256 / -0.0949 | 0.0819 / 0.0850 / 0.0829 / +0.0010 | 0.0703 / 0.0810 / 0.0746 / +0.0043 | 0.2767 / 0.0968 / 0.1611 / -0.1156 | 0.0859 |  |
| mask_only | scattered | 0.6458 / 0.5014 / 0.5039 / -0.1419 | 0.0955 / 0.0960 / 0.0944 / -0.0011 | 0.0706 / 0.0780 / 0.0730 / +0.0024 | 0.0325 / 0.0592 / 0.0547 / +0.0222 | 0.0874 |  |
| observed_neighbor | bridge_persistent | 0.5815 / 0.5613 / 0.5613 / -0.0202 | 0.0791 / 0.0942 / 0.0942 / +0.0151 | 0.1690 / 0.1528 / 0.1528 / -0.0162 | 0.4787 / 0.4534 / 0.4534 / -0.0253 | 0.1725 |  |
| observed_neighbor | degree_matched | 0.6007 / 0.5573 / 0.5573 / -0.0434 | 0.0845 / 0.0912 / 0.0912 / +0.0067 | 0.1864 / 0.1393 / 0.1393 / -0.0471 | 0.4799 / 0.4546 / 0.4546 / -0.0253 | 0.1528 |  |
| observed_neighbor | full | 0.8872 / 0.8235 / 0.8235 / -0.0637 | 0.2741 / 0.2630 / 0.2630 / -0.0111 | 0.3789 / 0.3270 / 0.3270 / -0.0519 | 0.0000 / 0.0000 / 0.0000 / +0.0000 | 0.3686 |  |
| observed_neighbor | random_persistent | 0.7214 / 0.6947 / 0.6947 / -0.0267 | 0.1293 / 0.1398 / 0.1398 / +0.0105 | 0.2388 / 0.2184 / 0.2184 / -0.0204 | 0.3675 / 0.3555 / 0.3555 / -0.0120 | 0.2398 |  |
| observed_neighbor | region_persistent | 0.6574 / 0.6078 / 0.6078 / -0.0496 | 0.1144 / 0.1156 / 0.1156 / +0.0012 | 0.2270 / 0.1764 / 0.1764 / -0.0506 | 0.3233 / 0.2909 / 0.2909 / -0.0324 | 0.2021 |  |
| observed_neighbor | scattered | 0.7540 / 0.7197 / 0.7197 / -0.0343 | 0.1859 / 0.2009 / 0.2009 / +0.0150 | 0.3173 / 0.2851 / 0.2851 / -0.0322 | 0.0409 / 0.0451 / 0.0451 / +0.0042 | 0.3203 |  |
| prevalence | bridge_persistent | 0.6010 / 0.5000 / 0.5000 / -0.1010 | 0.0758 / 0.0762 / 0.0762 / +0.0004 | 0.0883 / 0.0607 / 0.0607 / -0.0276 | 0.3131 / 0.1457 / 0.1457 / -0.1674 | 0.0680 |  |
| prevalence | degree_matched | 0.6028 / 0.5000 / 0.5000 / -0.1028 | 0.0763 / 0.0760 / 0.0760 / -0.0003 | 0.0856 / 0.0613 / 0.0613 / -0.0243 | 0.3263 / 0.1448 / 0.1448 / -0.1815 | 0.0690 |  |
| prevalence | full | 0.6543 / 0.5000 / 0.5000 / -0.1543 | 0.0973 / 0.0973 / 0.0973 / -0.0000 | 0.1484 / 0.0743 / 0.0743 / -0.0741 | 0.0000 / 0.0000 / 0.0000 / +0.0000 | 0.0873 |  |
| prevalence | random_persistent | 0.6104 / 0.5000 / 0.5000 / -0.1104 | 0.0781 / 0.0775 / 0.0775 / -0.0006 | 0.1038 / 0.0613 / 0.0613 / -0.0425 | 0.2272 / 0.1265 / 0.1265 / -0.1007 | 0.0677 |  |
| prevalence | region_persistent | 0.6049 / 0.5000 / 0.5000 / -0.1049 | 0.0767 / 0.0775 / 0.0775 / +0.0008 | 0.0904 / 0.0600 / 0.0600 / -0.0304 | 0.3194 / 0.1391 / 0.1391 / -0.1803 | 0.0679 |  |
| prevalence | scattered | 0.6503 / 0.5000 / 0.5000 / -0.1503 | 0.0953 / 0.0951 / 0.0951 / -0.0002 | 0.1463 / 0.0729 / 0.0729 / -0.0734 | 0.0228 / 0.0150 / 0.0150 / -0.0078 | 0.0861 |  |

Primary origins excluded from per-origin recall (zero positives), summed over test episodes:

| population | condition | excluded origins | of |
|---|---|---|---|
| all | bridge_persistent | 0 | 256 |
| all | degree_matched | 0 | 256 |
| all | full | 0 | 256 |
| all | random_persistent | 0 | 256 |
| all | region_persistent | 0 | 256 |
| all | scattered | 0 | 256 |
| candidate | bridge_persistent | 0 | 256 |
| candidate | degree_matched | 0 | 256 |
| candidate | full | 0 | 256 |
| candidate | random_persistent | 0 | 256 |
| candidate | region_persistent | 0 | 256 |
| candidate | scattered | 0 | 256 |

### Mask overlap

September's mask_overlap.json has keys ['episode', 'family', 'fraction_bridge_set_shared', 'graph']: the share of the bridge set also in the degree-matched control, per episode. It stores no host IDs, so a direct ours-vs-September comparison of bridge or region sets is not possible from it.

| episode | family | Sep bridge∩matched | ours bridge∩matched | ours bridge∩region |
|---|---|---|---|---|
| 3000_0 | BA | 0.7843 | 0.8333 | 0.5882 |
| 3000_1 | BA | 0.7941 | 0.8333 | 0.5882 |
| 3001_0 | SBM | 0.6176 | 0.6078 | 0.5196 |
| 3001_1 | SBM | 0.5980 | 0.6078 | 0.5196 |
| 3002_0 | BA | 0.7451 | 0.8039 | 0.6275 |
| 3002_1 | BA | 0.7647 | 0.8039 | 0.6275 |
| 3003_0 | SBM | 0.7549 | 0.7353 | 0.5784 |
| 3003_1 | SBM | 0.7843 | 0.7353 | 0.5784 |
| 3004_0 | BA | 0.8137 | 0.8529 | 0.5490 |
| 3004_1 | BA | 0.8039 | 0.8529 | 0.5490 |
| 3005_0 | SBM | 0.6667 | 0.7549 | 0.5392 |
| 3005_1 | SBM | 0.7157 | 0.7549 | 0.5392 |
| 3006_0 | BA | 0.8039 | 0.7745 | 0.5294 |
| 3006_1 | BA | 0.8137 | 0.7745 | 0.5294 |
| 3007_0 | SBM | 0.6961 | 0.6471 | 0.4902 |
| 3007_1 | SBM | 0.6961 | 0.6471 | 0.4902 |

Required rows present: fitted_mechanistic/full=yes, fitted_mechanistic/bridge_persistent=yes, boosted_trees/full=yes, boosted_trees/bridge_persistent=yes

### C05b notes

Stage 2 was rerun under Spec v1.0.1 into `results/stage2/run_b`; the old run is `run_a`. Stage 1 passes 13/13 and Stage 2 passes 7/7. The tables above show September / old / new.

**Remaining flags: the same 8 (all-host).**

| Flag | After v1.0.1 | Cause |
|---|---|---|
| observed_neighbor, 5 masked conditions, AUC +0.034 to +0.076 | Identical to C05 to 4 dp | The method has one input, fitted with a logistic model whose output rises with the neighbour count. More training rows change its coefficients but not the ranking. Known hosts already sat below every candidate (P > 0), so the -1 rule changes nothing. The gap comes from candidates with zero visible exposure (P ≈ 0.045) ranking above known negatives. That is a property of the all-host population, not of ties. On candidates the AUC gap is -0.02 to -0.05, not flagged there. |
| logistic, bridge / degree_matched / region, AUC -0.073 to -0.088 | Moved ≤ 0.004 | The training-row count now matches September (423,936). What remains is the feature set: 22 features against September's 32, which Spec v1.0.1 item 2 says not to reconstruct. |

**What v1.0.1 changed.** Training on 7..51 moved AUC and AP by at most 0.009 for every method. mask_only late_detection_share moved towards September (bridge 0.10 → 0.17, September 0.22).

**Recall.** No primary origin has zero positives (0 of 256 per condition), so the per-origin mean uses every origin. Per-origin recall is much closer to September than pooled recall. For full: fitted_mechanistic 0.3682 vs 0.3674, logistic 0.3624 vs 0.3737, boosted_trees 0.3672 vs 0.3820. This supports the C05 finding that September averaged per origin. Prevalence and mask_only remain below September (0.087 vs 0.148 on full), which the tie rule explains.

**(a) Mask overlap.** September's `mask_overlap.json` stores only the share of the bridge set that is also in the degree-matched control, with no host IDs. So our bridge and region sets cannot be compared with September's directly. On that shared measure, ours is 0.61–0.85 and September's is 0.60–0.81, and the per-graph differences run from -0.05 to +0.09. Cause: September's value differs between the two outbreaks of the same graph (for example 3005: 0.667 / 0.716), so its control set was drawn per outbreak. Ours is drawn once per graph (`stage1.py:57`, `matched_nodes` with the graph RNG), so both outbreaks share it. Our bridge set's share in the region set is 0.49–0.63; September has no counterpart.
