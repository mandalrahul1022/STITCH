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
