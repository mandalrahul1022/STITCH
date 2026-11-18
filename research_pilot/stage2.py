"""Fit on train, choose on validation AP, freeze choices, then evaluate test views once."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from classical import METHODS, Dynamics, adjacency, classifier, columns, design, predict
from metrics import (BOOTSTRAP_SEED, PRIMARY, csv_write, evaluate, finite_mean, reliability,
                     summaries, tie_keys)
from pilot import EVENT_ORIGINS, PRIMARY_ORIGINS, labels
from stage1 import FAMILIAR, HELD_OUT, write_json

STAGE1_TESTS = ("T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T09", "T10",
                "T10b", "T10c", "T10d")


def views(data, manifest, split, origins):
    episodes = {r["id"]: r for r in manifest["episodes"] if r["split"] == split}
    graphs = {}
    for v in manifest["views"]:
        if v["episode"] not in episodes:
            continue
        if split != "test" and v["condition"] not in FAMILIAR:
            raise ValueError("held-out mask present in model development")
        e = episodes[v["episode"]]
        g = e["graph"]
        if g not in graphs:
            with np.load(data / f"graph_{g}.npz") as f:
                graphs[g] = adjacency(f["edges"])
        with np.load(data / v["file"]) as f:
            x = f["x"]
        with np.load(data / e["file"]) as f:
            first = f["first"]  # evaluation/training labels only; never supplied for prediction
        yield SimpleNamespace(ep=e, condition=v["condition"], adj=graphs[g], x=x, first=first,
                              matrix=design(x, origins, graphs[g]))


def validation_ap(name, model, valid):
    # Average condition AP within episode, then episode AP within graph.
    graphs = {}
    for v in valid:
        s = predict(name, model, v.x, PRIMARY_ORIGINS, v.adj, v.matrix)
        y = np.array([labels(v.first, t) for t in PRIMARY_ORIGINS]).ravel()
        ap = average_precision_score(y, s.ravel()) if y.any() else np.nan
        graphs.setdefault(v.ep["graph"], {}).setdefault(v.ep["id"], []).append(ap)
    return float(finite_mean([finite_mean([finite_mean(c) for c in eps.values()])
                              for eps in graphs.values()]))


def fit(data, manifest, out):
    train = list(views(data, manifest, "train", PRIMARY_ORIGINS))
    valid = list(views(data, manifest, "val", PRIMARY_ORIGINS))
    x = np.concatenate([v.matrix.reshape(-1, v.matrix.shape[-1]) for v in train])
    y = np.concatenate([np.array([labels(v.first, t) for t in PRIMARY_ORIGINS]).ravel()
                        for v in train])
    models = {"prevalence": float(y.mean()), "fitted_mechanistic": Dynamics.fit(
        (v.x, v.adj) for v in train if v.condition == "full")}
    grid = {"degree": [1.0], "observed_neighbor": [1.0], "mask_only": [1.0],
            "logistic": [0.1, 1.0], "boosted_trees": [7, 15]}
    selection = {}
    for name in METHODS:
        if name in models:
            selection[name] = {"configuration": "fixed", "val_ap": validation_ap(name, models[name], valid)}
            continue
        candidates = []
        best_ap = -np.inf
        for setting in grid[name]:
            model = classifier(name, setting).fit(x[:, columns(name)], y)
            ap = validation_ap(name, model, valid)
            candidates.append(dict(setting=setting, val_ap=ap))
            if ap > best_ap:
                models[name], best_ap, best = model, ap, setting
        selection[name] = dict(selected=best, grid=candidates)
    write_json(out / "mechanistic_parameters.json", asdict(models["fitted_mechanistic"]))
    # Save standard sklearn objects and primitive process parameters for independent reuse.
    joblib.dump({**models, "fitted_mechanistic": asdict(models["fitted_mechanistic"])},
                out / "classical_models.joblib", compress=3)
    write_json(out / "validation_selection.json", dict(
        train_graphs=sorted({v.ep["graph"] for v in train}),
        validation_graphs=sorted({v.ep["graph"] for v in valid}),
        conditions=FAMILIAR, primary_origins=PRIMARY_ORIGINS, models=selection,
        seeds={"tie_break": "[outbreak_seed, origin_t, 773]",
               "bootstrap": BOOTSTRAP_SEED, "boosted_trees_random_state": 1701},
        data_manifest_sha256=hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest(),
        model_file_sha256=hashlib.sha256((out / "classical_models.joblib").read_bytes()).hexdigest()))
    return models


def evaluate_test(data, manifest, out, models):
    rows, diagnostic = [], []
    pooled = {}  # (population, condition, model) -> primary-origin targets and scores
    pred_dir = out / "predictions"
    pred_dir.mkdir()
    for view in views(data, manifest, "test", EVENT_ORIGINS):
        x, e = view.x, view.ep
        known = x[list(EVENT_ORIGINS), :, 6].astype(bool)
        keys = tie_keys(e["seed"], len(view.first))
        y_all = np.array([labels(view.first, t) for t in EVENT_ORIGINS])
        scores = {}
        for name in METHODS:
            scores[name] = predict(name, models[name], x, EVENT_ORIGINS, view.adj, view.matrix)
            if np.any(scores[name][known] != 0):
                raise ValueError("known host received a nonzero forecast")
            for pop, allowed in (("all", np.ones_like(known)), ("candidate", ~known)):
                row = dict(graph=e["graph"], episode=e["id"], beta=e["beta"], condition=view.condition,
                           model=name, population=pop)
                rows.append({**row, **evaluate(scores[name], view.first, allowed, keys)})
                keep = allowed & PRIMARY[:, None]
                bucket = pooled.setdefault((pop, view.condition, name), ([], []))
                bucket[0].append(y_all[keep])
                bucket[1].append(scores[name][keep])
        np.savez_compressed(pred_dir / f"{e['id']}_{view.condition}.npz", **scores)
        if view.condition == "bridge_persistent":
            target = np.array([labels(view.first, t) for t in PRIMARY_ORIGINS])
            prior_ei = np.maximum.accumulate((x[:, :, 1] + x[:, :, 2]) > 0, axis=0)
            for rule, eligible in (("E/I", ~prior_ei[list(PRIMARY_ORIGINS)]),
                                   ("E/I/P", ~known[PRIMARY])):
                if (target & ~eligible).any():
                    raise ValueError("known-prior exclusion removed a future event")
                diagnostic.append(dict(graph=e["graph"], episode=e["id"], rule=rule,
                                       positives=int(target[eligible].sum()),
                                       eligible=int(eligible.sum()), prevalence=float(target[eligible].mean())))
    summaries(rows, out, METHODS, FAMILIAR + HELD_OUT)
    # Diagnostic reliability table: primary host-origins pooled over all test episodes
    # (not graph-weighted). Per-episode ECE in the summaries uses the same binning.
    csv_write(out / "reliability.csv", [
        dict(population=pop, condition=cond, model=name, **r)
        for (pop, cond, name), (ys, ps) in pooled.items()
        for r in reliability(np.concatenate(ys), np.concatenate(ps))])
    csv_write(out / "bridge_prevalence_by_episode.csv", diagnostic)
    diag = {}
    for rule in ("E/I", "E/I/P"):
        subset = [r for r in diagnostic if r["rule"] == rule]
        pos, total = (sum(r[col] for r in subset) for col in ("positives", "eligible"))
        by_graph = {}
        for r in subset:
            by_graph.setdefault(r["graph"], []).append(r["prevalence"])
        diag[rule] = dict(graph_macro=float(np.mean([np.mean(v) for v in by_graph.values()])),
                          pooled=pos / total, positives=pos, eligible=total)
    write_json(out / "bridge_prevalence.json", diag)


def run(data, out):
    manifest = json.loads((data / "manifest.json").read_text())
    # Written by: pytest research_pilot/tests --stitch-data <data> --stitch-archive <zip>
    checks = {t["id"]: t["status"] for t in json.loads((data / "verification.json").read_text())["tests"]}
    missing = [t for t in STAGE1_TESTS if checks.get(t) != "PASS"]
    if missing or manifest["config"]["confirmation"] != "E-I-P":
        raise ValueError(f"Stage 1 verified E/I/P data required; not PASS: {missing}")
    out.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        models = fit(data, manifest, out)
        print("Training and validation complete. Choices frozen before opening test views.", flush=True)
        evaluate_test(data, manifest, out, models)
    print("Classical evaluation complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)
