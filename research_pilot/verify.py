"""Audit Stage 1 fixtures and serialized trajectories without evaluating any predictor."""
import argparse
import inspect
import json
from pathlib import Path
import unittest

import numpy as np

from pilot import (CHANNELS, S, E, I, P, Process, advance, features, labels,
                   matched_nodes, visibility, visible_evidence, window)
from topology import betweenness, connected_region, neighbors, edge_hash
from pilot import truth_hash
from zipfile import ZipFile


def tick_fixture(progression, completion):
    state = np.array([I, S], dtype=np.uint8)
    first = np.array([0, -1])
    queued = np.zeros(2, dtype=bool)
    ref = np.ones(2, dtype=bool)
    process = Process(1, 0, progression, 1, completion)
    advance(state, first, queued, ref, np.array([[0, 1]]), 1, process,
            np.random.default_rng(23))
    return state, first


def observed_fixture():
    states = np.tile(np.arange(4), (57, 1))
    visible = np.ones_like(states, dtype=bool)
    visible[12:44, 1:3] = False
    degrees = np.array([0, 0.25, 0.5, 1])
    return states, visible, degrees


class Invariants(unittest.TestCase):
    def test_01_distinct_graph_hashes_and_disjoint_splits(self):
        splits = {"train": set(range(1000, 1012)), "val": set(range(2000, 2004)),
                  "test": set(range(3000, 3008))}
        hashes = []
        rows = self.manifest["graphs"]
        for split, ids in splits.items():
            self.assertEqual({r["seed"] for r in rows if r["split"] == split}, ids)
        for r in rows:
            with np.load(self.data / f"graph_{r['seed']}.npz") as f:
                hashes.append(edge_hash(f["edges"]))
            self.assertEqual(hashes[-1], r["hash"])
        self.assertEqual(len(rows), 24)
        self.assertEqual(len(set(hashes)), 24)
        with ZipFile(self.archive) as z:
            old = json.loads(z.read("results/manifest.json"))
        self.assertEqual(sorted(hashes), sorted(r["hash"] for r in old["graphs"]))
        def identifiers(rows):
            return sorted((r["id"], r["graph"], r["split"], r["beta"], r["seed"]) for r in rows)
        self.assertEqual(identifiers(old["episodes"]), identifiers(self.manifest["episodes"]))
        for row in self.manifest["episodes"]:
            with np.load(self.data / row["file"]) as f:
                self.assertEqual(truth_hash(f), row["truth_hash"])
        for ep in self.manifest["episodes"]:
            conds = [v["condition"] for v in self.manifest["views"] if v["episode"] == ep["id"]]
            expected = ["full", "scattered", "random_persistent"]
            if ep["split"] == "test":
                expected += ["bridge_persistent", "degree_matched", "region_persistent"]
            self.assertCountEqual(conds, expected)

    def test_02_capture_before_progression_or_patch(self):
        for progression, completion, final in [(0, 0, E), (1, 0, I), (1, 1, P)]:
            state, first = tick_fixture(progression, completion)
            self.assertEqual((state[1], first[1]), (final, 1))

    def test_03_hidden_truth_invariant(self):
        states, visible, degree = observed_fixture()
        before = features(visible_evidence(states, visible), degree, (E, I, P))
        states[~visible] = (states[~visible] + 1) % 4
        after = features(visible_evidence(states, visible), degree, (E, I, P))
        np.testing.assert_array_equal(before, after)
        self.assertFalse(before[~visible][:, :4].any())

    def test_04_prefix_only(self):
        states, visible, degree = observed_fixture()
        evidence = visible_evidence(states, visible)
        x = features(evidence, degree, (E, I, P))
        for t in range(7, 57):
            prefix = features(evidence[:t + 1], degree, (E, I, P))
            np.testing.assert_array_equal(window(x, t), prefix[-8:])
        self.assertEqual(x[43, 1, 5], 1)
        self.assertEqual(x[44, 1, 5], 0)
        self.assertEqual(x[43, 1, 6], 1)

    def test_05_prior_events_excluded(self):
        first = np.array([-1, 0, 7, 10, 11, 15, 16])
        np.testing.assert_array_equal(labels(first, 10), [0, 0, 0, 0, 1, 1, 0])

    def test_06_equal_mask_budgets(self):
        # The star fixture also gives an analytic centrality ranking.
        adj = neighbors(256, np.array([[0, v] for v in range(1, 256)]))
        scores = betweenness(adj)
        self.assertTrue(scores[0] > 0 and np.all(scores[1:] == 0))
        bridge = np.argsort(-scores, kind="stable")[:102]
        rng = np.random.default_rng(7)
        degree = np.array([len(n) for n in adj])
        matched = matched_nodes(degree, bridge, rng)
        region = connected_region(adj, 88, 102)
        self.assertIn(0, region)
        for condition, selected in [("full", None), ("scattered", None),
                                    ("random_persistent", None), ("bridge_persistent", bridge),
                                    ("degree_matched", matched), ("region_persistent", region)]:
            visible = visibility(57, 256, condition, rng, persistent_set=selected)
            expected = np.zeros(57)
            if condition != "full":
                expected[12:44] = 102
            np.testing.assert_array_equal((~visible).sum(axis=1), expected)
            if condition == "scattered":
                self.assertFalse(np.array_equal(visible[12], visible[13]))
            elif condition != "full":
                self.assertTrue((visible[12:44] == visible[12]).all())

        for row in self.manifest["views"]:
            with np.load(self.data / row["file"]) as f:
                mask = f["x"][:, :, 4] == 0
            expected = np.zeros(57)
            if row["condition"] != "full":
                expected[12:44] = 102
            np.testing.assert_array_equal(mask.sum(axis=1), expected)

    def test_07_degree_balance(self):
        degrees = np.repeat([1, 2, 7, 10], 64)
        bridge = np.arange(60, 162)
        control = matched_nodes(degrees, bridge, np.random.default_rng(55))
        self.assertEqual(len(np.unique(control)), len(bridge))
        np.testing.assert_array_equal(sorted(degrees[bridge]), sorted(degrees[control]))

        for row in self.manifest["graphs"]:
            if row["split"] == "test":
                with np.load(self.data / f"graph_{row['seed']}.npz") as f:
                    degree = np.bincount(f["edges"].ravel(), minlength=256)
                    np.testing.assert_array_equal(sorted(degree[f["degree_matched"]]),
                                                  sorted(degree[f["bridge_persistent"]]))

    def test_08_tiny_graph_within_tick(self):
        # Force progression and patch completion to exercise the within-tick event.
        state = np.array([I, S], dtype=np.uint8)
        first = np.array([0, -1])
        reference = np.array([False, True])
        queued = np.zeros(2, dtype=bool)
        advance(state, first, queued, reference, np.array([[0, 1]]), 4,
                Process(1, 0, 1, 1, 1), np.random.default_rng(4))
        np.testing.assert_array_equal(state, [I, P])
        np.testing.assert_array_equal(first, [0, 4])
        self.assertFalse(queued.any())
        self.assertTrue(labels(first, 3)[1])
        self.assertFalse(labels(first, 4)[1])

    def test_09_horizon_edges(self):
        for t in (7, 13, 43, 51):
            np.testing.assert_array_equal(labels(np.array([t, t + 5, t + 6]), t),
                                          [False, True, False])

    def test_10_input_schema_and_clock_queue_exclusion(self):
        self.assertEqual(tuple(inspect.signature(features).parameters),
                         ("evidence", "normalized_log_degree", "compromised_codes"))
        self.assertEqual(len(CHANNELS), 8)
        state, visible, degree = observed_fixture()
        x = features(visible_evidence(state, visible), degree, (E, I, P))
        # Independent channel oracle reconstructs the complete tensor at each origin.
        for t in range(len(x)):
            for host in range(4):
                prefix = np.flatnonzero(visible[:t + 1, host])
                code = state[t, host] if visible[t, host] else -1
                prior = np.isin(state[prefix, host], [E, I, P]).any()
                age = min(32, t - prefix[-1]) if prefix.size else 32
                expected = [code == s for s in (S, E, I, P)]
                expected += [visible[t, host], age / 32, prior, degree[host]]
                np.testing.assert_array_equal(x[t, host], expected)


        records = {row["id"]: row for row in self.manifest["episodes"]}
        for view in self.manifest["views"]:
            with np.load(self.data / view["file"]) as f:
                observed = f["x"]
            with np.load(self.data / records[view["episode"]]["file"]) as f:
                truth = f["states"]
            with np.load(self.data / f"graph_{view['graph']}.npz") as f:
                degree = np.bincount(f["edges"].ravel(), minlength=256)
            visible = observed[:, :, 4] == 1
            rebuilt = features(visible_evidence(truth, visible),
                               np.log1p(degree) / np.log1p(degree.max()), (E, I, P))
            np.testing.assert_array_equal(rebuilt, observed)
            confirmed = np.maximum.accumulate(visible & (truth != S), axis=0)
            np.testing.assert_array_equal(confirmed, observed[:, :, 6])

    def test_11_P_only_from_reference_observed_I_in_queue(self):
        count = same_tick = 0
        for record in self.manifest["episodes"]:
            with np.load(self.data / record["file"]) as f:
                states, qq = f["states"], f["queues"]
                pre, queued, ref = f["pre_patch_state"][1:], f["pre_patch_queue"][1:], f["reference"]
                patched = (states[:-1] != P) & (states[1:] == P)
                np.testing.assert_array_equal(states[0] == I, f["first"] == 0)
                self.assertFalse((states[0] == P).any())
                self.assertFalse((np.diff(states.astype(int), axis=0) < 0).any())
                self.assertTrue((pre[queued] == I).all())
                self.assertFalse((queued & ~ref).any())
                self.assertFalse((qq[:-1] & ~queued).any())
                self.assertFalse((patched & ~queued).any())
                np.testing.assert_array_equal(queued & ~patched, qq[1:])
                np.testing.assert_array_equal(states[1:], np.where(patched, P, pre))
                entrants = queued & ~qq[:-1]
                eligible = ref & (pre == I) & ~qq[:-1]
                np.testing.assert_array_equal(entrants.sum(1), np.rint(0.1 * eligible.sum(1)))
                count += int(patched.sum())
                same_tick += int((patched & entrants).sum())
        self.assertGreater(count, 0)
        self.audit_counts.update(patches=count, queued_and_patched_same_tick=same_tick)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    Invariants.data, Invariants.archive = args.data, args.archive
    Invariants.manifest = json.loads((args.data / "manifest.json").read_text())
    Invariants.audit_counts = {}
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Invariants)
    names = [test._testMethodName for test in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failed = {test._testMethodName for test, _ in result.errors + result.failures}
    (args.data / "verification.json").write_text(json.dumps({
        "tests": {name: ("FAIL" if name in failed else "PASS") for name in names},
        "counts": Invariants.audit_counts,
    }, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
