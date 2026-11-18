"""Stage 2 fixture checks. Optionally audit the materialized test output."""
import argparse
import csv
import json
from pathlib import Path
import unittest
import numpy as np

from classical import METHODS, Dynamics, adjacency, columns, design
from metrics import PRIMARY, evaluate, ranks, tie_keys
from pilot import EVENT_ORIGINS, labels, features
from stage1 import FAMILIAR, HELD_OUT, write_json


class ClassicalChecks(unittest.TestCase):
    def test_01_future_event_and_earliest_positive_lead(self):
        origins = np.array(EVENT_ORIGINS)
        first = np.full(24, -1)
        first[[0, 1]] = [18, 0]
        s = np.zeros((len(origins), 24))
        allowed = np.ones_like(s, dtype=bool)
        keys = np.tile(np.arange(24), (len(origins), 1))
        got = evaluate(s, first, allowed, keys)
        self.assertEqual(got["unique_event_recall"], 1)
        self.assertEqual(got["lead_time"], 5)
        self.assertEqual(got["recall_at_budget"], 1)
        self.assertEqual(got["positive_host_origins"], 3)
        self.assertEqual(got["eligible_host_origins"], 16 * 24)
        self.assertAlmostEqual(got["precision_at_budget"], 3 / (13 * 16))
        past = sum(1 + int(t >= 18) for t in origins[PRIMARY])
        self.assertAlmostEqual(got["late_detection_share"], past / (13 * 16))

    def test_02_candidate_filter_ties_and_budget(self):
        s = np.zeros(30)
        allowed = np.arange(30) % 2 == 0
        key = tie_keys(45, 30)[0]
        chosen = ranks(s, allowed, key)
        self.assertEqual(len(chosen), 13)
        self.assertTrue(allowed[chosen].all())
        np.testing.assert_array_equal(chosen, ranks(s, allowed, tie_keys(45, 30)[0]))
        self.assertEqual(len(ranks(s, np.arange(30) < 3, key)), 3)

    def test_03_mask_only_and_causal_model_inputs(self):
        n = 12
        adj = adjacency(np.array([[k, k+1] for k in range(n-1)]), nodes=n)
        obs = np.zeros((30, n), dtype=int)
        obs[10:24, :4] = -1
        x = features(obs, np.linspace(0, 1, n), (1, 2, 3))
        modified = obs.copy()
        modified[modified == 0] = 2
        xx = features(modified, np.linspace(0, 1, n), (1, 2, 3))
        a, b = design(x, [13], adj), design(xx, [13], adj)
        np.testing.assert_array_equal(a[:, :, columns("mask_only")], b[:, :, columns("mask_only")])
        dyn = Dynamics(0.07, 0.5, 0.08, [.7, .1, .1, .1])
        score = dyn.score(x, 13, adj)
        x[14:] = 1
        np.testing.assert_array_equal(dyn.score(x, 13, adj), score)
        np.testing.assert_array_equal(design(x, [13], adj), a)

    def test_04_filter_mass_and_observed_state(self):
        adj = adjacency(np.array([[0, 1], [1, 2]]), nodes=3)
        dyn = Dynamics(.08, .5, .1, [.7, .1, .1, .1])
        belief = np.eye(4)[[0, 2, 3]]
        after = dyn.advance(belief, adj)
        np.testing.assert_allclose(after.sum(1), 1)
        self.assertGreater(after[0, 1], 0)
        self.assertTrue((after >= 0).all())
        x = features(np.tile([0, 2, 3], (8, 1)), np.ones(3), (1, 2, 3))
        pred = dyn.score(x, 7, adj)
        self.assertGreater(pred[0], 0)
        np.testing.assert_array_equal(pred[1:], [0, 0])


class OutputChecks(unittest.TestCase):
    def test_05_frozen_splits_and_complete_score_grid(self):
        out, data = self.out, self.data
        lock = json.loads((out / "validation_selection.json").read_text())
        self.assertEqual(lock["train_graphs"], list(range(1000, 1012)))
        self.assertEqual(lock["validation_graphs"], list(range(2000, 2004)))
        self.assertEqual(lock["conditions"], list(FAMILIAR))
        with (out / "episode_metrics.csv").open() as f:
            rows = list(csv.DictReader(f))
        unique = {(r["episode"], r["model"], r["population"], r["condition"]) for r in rows}
        expected = {(f"{g}_{j}", m, p, c) for g in range(3000, 3008) for j in (0, 1)
                    for m in METHODS for p in ("all", "candidate") for c in FAMILIAR + HELD_OUT}
        self.assertEqual(unique, expected)
        self.assertEqual(len(unique), len(rows))
        with (out / "summary.csv").open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 84)

    def test_06_all_saved_scores_respect_confirmation(self):
        manifest = json.loads((self.data / "manifest.json").read_text())
        episodes = {e["id"]: e for e in manifest["episodes"] if e["split"] == "test"}
        for v in manifest["views"]:
            if v["episode"] not in episodes:
                continue
            with np.load(self.data / v["file"]) as f:
                known = f["x"][list(EVENT_ORIGINS), :, 6].astype(bool)
            path = self.out / "predictions" / (v["episode"]+"_"+v["condition"]+".npz")
            with np.load(path) as f:
                self.assertCountEqual(f.files, METHODS)
                for name in METHODS:
                    s = f[name]
                    self.assertEqual(s.shape, known.shape)
                    self.assertTrue(np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all())
                    self.assertTrue((s[known] == 0).all())

    def test_07_bridge_diagnostic_and_primary_population(self):
        with (self.out / "bridge_prevalence_by_episode.csv").open() as f:
            rows = list(csv.DictReader(f))
        with (self.out / "episode_metrics.csv").open() as f:
            metrics = {r["episode"]: r for r in csv.DictReader(f) if r["condition"] == "bridge_persistent"
                       and r["population"] == "candidate" and r["model"] == "prevalence"}
        episodes = sorted(metrics)
        self.assertEqual(len(rows), 2 * len(episodes))
        for ep in episodes:
            d = {r["rule"]: r for r in rows if r["episode"] == ep}
            self.assertEqual(d["E/I/P"]["positives"], d["E/I"]["positives"])
            self.assertLessEqual(int(d["E/I/P"]["eligible"]), int(d["E/I"]["eligible"]))
            self.assertEqual(d["E/I/P"]["eligible"], metrics[ep]["eligible_host_origins"])
            self.assertEqual(d["E/I/P"]["positives"], metrics[ep]["positive_host_origins"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ClassicalChecks)
    if args.out:
        OutputChecks.data, OutputChecks.out = args.data, args.out
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(OutputChecks))
    names = [t._testMethodName for t in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failed = {t._testMethodName for t, _ in result.errors + result.failures}
    if args.out:
        write_json(args.out / "verification.json", {n: ("FAIL" if n in failed else "PASS") for n in names})
    raise SystemExit(0 if result.wasSuccessful() else 1)
