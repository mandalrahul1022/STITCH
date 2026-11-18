"""Stage 1 data and label invariants, spec section 4.1 (T01 to T10c) plus T10d.

T01-T10 and T10d run the inherited checks in verify.py unchanged (Invariants test methods) and add
evidence strings. T10b and T10c are new in C03.
"""
import numpy as np
import pytest

import verify
from archive_graphs import verified_graphs
from pilot import E, I, P, features, simulate, truth_hash, visibility, visible_evidence
from stage1 import FAMILIAR, HELD_OUT
from topology import edge_hash

GRAPH_SETS = ("bridge_persistent", "degree_matched", "region_persistent")


@pytest.fixture(scope="session")
def inv(data_dir, archive, manifest):
    """Configure verify.Invariants the way verify.py's command line does."""
    verify.Invariants.data, verify.Invariants.archive = data_dir, archive
    verify.Invariants.manifest = manifest
    verify.Invariants.audit_counts = {}
    return lambda name: getattr(verify.Invariants(name), name)()


def split_counts(manifest):
    out = {}
    for g in manifest["graphs"]:
        out[g["split"]] = out.get(g["split"], 0) + 1
    return out


def test_T01_distinct_graph_hashes_and_disjoint_splits(inv, manifest, evidence):
    inv("test_01_distinct_graph_hashes_and_disjoint_splits")
    hashes = {g["hash"] for g in manifest["graphs"]}
    evidence.append(f"{len(manifest['graphs'])} graphs {split_counts(manifest)}, {len(hashes)} distinct "
                    f"edge hashes equal to the September manifest; {len(manifest['episodes'])} episode truth "
                    f"hashes re-verified from episode files")


def test_T02_event_survives_within_tick(inv, evidence):
    inv("test_02_capture_before_progression_or_patch")
    evidence.append("fixture S->E, S->E->I, S->E->I->P in one tick: C(v)=1 recorded in all three")


def test_T03_hidden_state_changes_no_feature(inv, evidence):
    inv("test_03_hidden_truth_invariant")
    evidence.append("57x4 fixture, hidden cells' true state shifted by one: all 8 channels identical; "
                    "hidden cells have channels 0-3 all zero")


def test_T04_observations_prefix_only(inv, evidence):
    inv("test_04_prefix_only")
    evidence.append("features from the prefix [0..t] equal the full-run window at t for t=7..56")


def test_T05_labels_exclude_prior_compromise(inv, evidence):
    inv("test_05_prior_events_excluded")
    evidence.append("C=[-1,0,7,10,11,15,16] at t=10 gives y=[0,0,0,0,1,1,0]")


def test_T06_equal_missing_budgets(inv, manifest, evidence):
    inv("test_06_equal_mask_budgets")
    n = sum(v["condition"] != "full" for v in manifest["views"])
    evidence.append(f"fixture: all 5 missing conditions hide 102 hosts per tick for 12<=t<44 only; "
                    f"data: {n} missing-condition views each hide 102 per tick in [12,44) and 0 elsewhere")


def test_T07_degree_matched_histogram(inv, manifest, evidence):
    inv("test_07_degree_balance")
    overlaps = [g["matched_bridge_overlap"] for g in manifest["graphs"] if g["split"] == "test"]
    evidence.append(f"fixture and all {len(overlaps)} test graphs: sorted degrees of control == bridge; "
                    f"overlap with bridge set per graph {overlaps} of 102 "
                    f"(mean {np.mean(overlaps) / 102:.3f})")


def test_T08_tiny_graph_within_tick(inv, evidence):
    inv("test_08_tiny_graph_within_tick")
    evidence.append("2-host graph, forced transmission/progression/completion at tick 4: host 1 ends P "
                    "with C=4, y(t=3)=1, y(t=4)=0")


def test_T09_horizon_boundaries(inv, evidence):
    inv("test_09_horizon_edges")
    evidence.append("t in {7,13,43,51}: C=t negative, C=t+5 positive, C=t+6 negative")


def test_T10_no_queue_or_tick_in_inputs(inv, manifest, evidence):
    inv("test_10_input_schema_and_clock_queue_exclusion")
    evidence.append(f"features() takes only (evidence, normalized_log_degree, compromised_codes); "
                    f"independent channel oracle matches every cell; all {len(manifest['views'])} stored "
                    f"views rebuilt exactly from visible truth + degree")


def test_T10b_truth_identical_across_mask_conditions(data_dir, manifest, evidence):
    """Every view of an episode is rebuilt from that episode's single stored truth (hash checked)
    and the mask regenerated from its recorded mask seed."""
    episodes = {e["id"]: e for e in manifest["episodes"]}
    graphs, truths, rebuilt, per_episode = {}, {}, 0, {}
    for v in manifest["views"]:
        e = episodes[v["episode"]]
        if e["id"] not in truths:
            with np.load(data_dir / e["file"]) as f:
                episode = {k: f[k] for k in f.files}
            assert truth_hash(episode) == e["truth_hash"]
            truths[e["id"]] = episode["states"]
        if v["graph"] not in graphs:
            with np.load(data_dir / f"graph_{v['graph']}.npz") as f:
                graphs[v["graph"]] = {k: f[k] for k in f.files}
        g = graphs[v["graph"]]
        degree = np.bincount(g["edges"].ravel(), minlength=256)
        visible = visibility(57, 256, v["condition"], np.random.default_rng(v["mask_seed"]),
                             persistent_set=g.get(v["condition"]))
        with np.load(data_dir / v["file"]) as f:
            x = f["x"]
        np.testing.assert_array_equal(visible, x[:, :, 4] == 1)
        expected = features(visible_evidence(truths[e["id"]], visible),
                            np.log1p(degree) / np.log1p(degree.max()), (E, I, P))
        np.testing.assert_array_equal(expected, x)
        per_episode.setdefault(e["id"], set()).add(v["condition"])
        rebuilt += 1
    for e in manifest["episodes"]:
        wanted = set(FAMILIAR) | (set(HELD_OUT) if e["split"] == "test" else set())
        assert per_episode[e["id"]] == wanted
    six = sum(len(c) == 6 for c in per_episode.values())
    evidence.append(f"{rebuilt}/{len(manifest['views'])} views: mask regenerated from mask_seed and features "
                    f"rebuilt from the episode's one stored truth; {len(truths)} truth hashes verified; "
                    f"{six} test episodes carry all six conditions on the same truth hash")


def test_T10c_regenerate_from_manifest(archive, data_dir, manifest, evidence):
    """Graph hashes from the archive replay and truth hashes from re-simulation match the manifest."""
    rows, graphs = verified_graphs(archive)
    by_seed = {g["seed"]: g for g in manifest["graphs"]}
    for seed, edges in graphs.items():
        assert edge_hash(edges) == by_seed[seed]["hash"]
        with np.load(data_dir / f"graph_{seed}.npz") as f:
            np.testing.assert_array_equal(f["edges"], edges)
    for e in manifest["episodes"]:
        episode = simulate(256, graphs[e["graph"]], e["seed"], e["beta"])
        assert truth_hash(episode) == e["truth_hash"], e["id"]
    evidence.append(f"{len(graphs)}/24 graph hashes reproduced from the archive replay; "
                    f"{len(manifest['episodes'])}/48 truth hashes reproduced by re-simulating with the "
                    f"recorded outbreak seed and beta")


def test_T10d_patch_only_from_queue(inv, evidence):
    inv("test_11_P_only_from_reference_observed_I_in_queue")
    c = verify.Invariants.audit_counts
    evidence.append(f"every transition to P was queued, reference-observed and in state I before completion; "
                    f"enqueue count == rint(0.10 x eligible) every tick; {c['patches']} completions, "
                    f"{c['queued_and_patched_same_tick']} queued and completed in the same tick")
