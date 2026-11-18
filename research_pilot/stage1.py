"""Generate new episodes on the 24 verified archived topologies."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from archive_graphs import verified_graphs
from pilot import (E, I, P, EVENT_ORIGINS, PRIMARY_ORIGINS, features, labels,
                   matched_nodes, simulate, truth_hash, visibility, visible_evidence)
from topology import betweenness, connected_region, neighbors


FAMILIAR = ("full", "scattered", "random_persistent")
HELD_OUT = ("bridge_persistent", "degree_matched", "region_persistent")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run(archive, out):
    rows, graphs = verified_graphs(archive)
    with ZipFile(archive) as z:
        old = json.loads(z.read("results/manifest.json"))["episodes"]
    seeds = {(e["graph"], e["beta"]): e for e in old}
    keys = {(r["seed"], b) for r in rows for b in (0.04, 0.09)}
    if set(seeds) != keys or len(old) != len(keys) or len({e["seed"] for e in old}) != len(keys):
        raise ValueError("incomplete or duplicate outbreak seeds in archive")
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    out.mkdir(parents=True)
    records = []
    observations = []
    confirm_codes = (E, I, P)
    for row in rows:
        seed = row["seed"]
        edges = graphs[seed]
        adj = neighbors(256, edges)
        degree = np.array([len(n) for n in adj])
        log_degree = np.log1p(degree) / np.log1p(degree.max())
        conditions = FAMILIAR
        fixed = {}
        if row["split"] == "test":
            conditions += HELD_OUT
            rng = np.random.default_rng([seed, 918])
            fixed["bridge_persistent"] = np.argsort(-betweenness(adj), kind="stable")[:102]
            fixed["degree_matched"] = matched_nodes(degree, fixed["bridge_persistent"], rng)
            fixed["region_persistent"] = connected_region(adj, int(rng.integers(256)), 102)
            bridge, control = fixed["bridge_persistent"], fixed["degree_matched"]
            np.testing.assert_array_equal(np.sort(degree[bridge]), np.sort(degree[control]))
            row["matched_bridge_overlap"] = int(np.intersect1d(bridge, control).size)
        np.savez_compressed(out / f"graph_{seed}.npz", edges=edges, **fixed)
        for j, beta in enumerate((0.04, 0.09)):
            record = seeds[(seed, beta)]
            if record["id"] != f"{seed}_{j}" or record["split"] != row["split"]:
                raise ValueError("archived episode identity mismatch")
            outbreak_seed = record["seed"]
            episode = simulate(256, edges, outbreak_seed, beta)
            filename = f"episode_{seed}_{j}.npz"
            np.savez_compressed(out / filename, **episode)
            first = episode["first"]
            records.append(dict(graph=seed, id=f"{seed}_{j}", split=row["split"],
                                seed=outbreak_seed, beta=beta, file=filename,
                                truth_hash=truth_hash(episode),
                                ever_compromised=int((first >= 0).sum()),
                                first_events_in_forecast_windows=int(np.logical_or.reduce(
                                    [labels(first, t) for t in EVENT_ORIGINS]).sum())))
            for k, condition in enumerate(conditions):
                mask_seed = [seed, j, k, 503]
                visible = visibility(57, 256, condition, np.random.default_rng(mask_seed),
                                     persistent_set=fixed.get(condition))
                counts = np.zeros(57, dtype=int)
                if condition != "full":
                    counts[12:44] = 102
                np.testing.assert_array_equal((~visible).sum(axis=1), counts)
                encoded = features(visible_evidence(episode["states"], visible),
                                   log_degree, confirm_codes)
                view_file = f"view_{seed}_{j}_{condition}.npz"
                np.savez_compressed(out / view_file, x=encoded)
                observations.append(dict(graph=seed, episode=f"{seed}_{j}", condition=condition,
                                         mask_seed=mask_seed, file=view_file))
    write_json(out / "manifest.json", {
        "graph_source": "September archived edge-list replay, freshly simulated episodes",
        "config": dict(nodes=256, ticks=56, history=8, horizon=5, initial_infected=3,
                       betas=[0.04, 0.09], volatility=0.20, progression=0.5,
                       queue_weight=0.5, reference_coverage=0.80, patch_rate=0.10,
                       completion=0.33, rewiring=0, hidden_nodes=102, outage=[12, 44],
                       primary_origins=PRIMARY_ORIGINS, event_origins=EVENT_ORIGINS,
                       training_masks=FAMILIAR, test_only_masks=HELD_OUT,
                       confirmation="E-I-P", rng="NumPy Generator PCG64",
                       outbreak_seed_source="results/manifest.json inside September ZIP", log_degree="log1p(degree)/log1p(max degree)",
                       count_window="distinct first events in the union of extended windows"),
        "graphs": rows, "episodes": records, "views": observations,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.archive, args.out)
