"""Episode metrics and shared graph-level uncertainty for the fixed Stage 2 benchmark."""
import csv
from itertools import combinations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pilot import EVENT_ORIGINS, PRIMARY_ORIGINS, labels

METRICS = ("auc", "ap", "precision_at_budget", "recall_at_budget", "recall_at_budget_per_origin",
           "late_detection_share", "unique_event_recall", "lead_time", "brier", "ece", "prevalence")
PRIMARY = np.isin(EVENT_ORIGINS, PRIMARY_ORIGINS)
RELIABILITY_BINS = 10
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 1702


def reliability(target, prob, bins=RELIABILITY_BINS):
    """Equal-mass bins over a stable sort of the scores: (count, mean score, event rate) per bin."""
    order = np.argsort(prob, kind="stable")
    rows = []
    for k, idx in enumerate(np.array_split(order, bins)):
        if idx.size:
            rows.append(dict(bin=k, n=int(idx.size), mean_score=float(prob[idx].mean()),
                             event_rate=float(target[idx].mean())))
    return rows


def ece(target, prob, bins=RELIABILITY_BINS):
    """Expected calibration error over equal-mass bins, weighted by bin size."""
    rows = reliability(target, prob, bins)
    total = sum(r["n"] for r in rows)
    return sum(r["n"] / total * abs(r["mean_score"] - r["event_rate"]) for r in rows) if total else np.nan


def ranks(scores, candidates, key):
    ids = np.flatnonzero(candidates)
    return ids[np.lexsort((key[ids], -scores[ids]))[:13]]


def tie_keys(seed, nodes):
    return np.array([np.random.default_rng([seed, t, 773]).permutation(nodes)
                     for t in EVENT_ORIGINS])


def ratio(a, b):
    return float(a / b) if b else np.nan


def evaluate(scores, first, allowed, keys):
    origins = np.array(EVENT_ORIGINS)
    y = np.array([labels(first, t) for t in origins])
    chosen = np.zeros_like(y)
    for j in range(len(origins)):
        chosen[j, ranks(scores[j], allowed[j], keys[j])] = True
    primary = allowed & PRIMARY[:, None]
    selected = chosen & PRIMARY[:, None]
    target, prob = y[primary], scores[primary]
    hits = (selected & y).sum()
    # Secondary recall (Spec v1.0.1): mean over primary origins of hits/positives, skipping
    # origins with no positive host; the skipped count is reported alongside.
    pos_j = (y & primary).sum(axis=1)[PRIMARY]
    hit_j = (selected & y).sum(axis=1)[PRIMARY]
    per_origin = hit_j[pos_j > 0] / pos_j[pos_j > 0]
    calibrated = np.clip(prob, 0, 1)  # known hosts carry KNOWN_SCORE < 0; treat as 0 for Brier/ECE
    warned = chosen & y
    detected = warned.any(axis=0)
    eligible = (y & allowed).any(axis=0)
    lead = first[detected] - origins[np.argmax(warned[:, detected], axis=0)]
    prior = (first >= 0) & (first <= origins[:, None])
    return dict(auc=roc_auc_score(target, prob) if np.unique(target).size == 2 else np.nan,
                ap=average_precision_score(target, prob) if target.any() else np.nan,
                precision_at_budget=ratio(hits, selected.sum()),
                recall_at_budget=ratio(hits, target.sum()),
                recall_at_budget_per_origin=float(per_origin.mean()) if per_origin.size else np.nan,
                late_detection_share=ratio((selected & prior).sum(), selected.sum()),
                unique_event_recall=ratio(detected.sum(), eligible.sum()),
                lead_time=float(lead.mean()) if lead.size else np.nan,
                brier=np.mean((calibrated - target)**2), ece=ece(target, calibrated), prevalence=target.mean(),
                origins_zero_positive=int((pos_j == 0).sum()),
                positive_host_origins=int(target.sum()), eligible_host_origins=int(target.size),
                eligible_unique_events=int(eligible.sum()), warned_unique_events=int(detected.sum()))


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(a, axis=0):
    a = np.asarray(a, dtype=float)
    count = np.isfinite(a).sum(axis=axis)
    return np.nansum(a, axis=axis) / np.where(count > 0, count, np.nan)


def summaries(rows, out, methods, conditions):
    """Same graph-index resamples for every method, condition and population."""
    graphs = sorted({r["graph"] for r in rows})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(len(graphs), size=(BOOTSTRAP_RESAMPLES, len(graphs)))
    graph_rows, point, intervals, differences = [], [], [], []
    for pop in ("all", "candidate"):
        for cond in conditions:
            estimates = {}
            for model in methods:
                subset = [r for r in rows if (r["population"], r["condition"], r["model"])
                          == (pop, cond, model)]
                info = dict(population=pop, condition=cond, model=model)
                cells = []
                for g in graphs:
                    episodes = [r for r in subset if r["graph"] == g]
                    if len(episodes) != 2:
                        raise ValueError("incomplete paired outbreak design")
                    values = finite_mean([[r[m] for m in METRICS] for r in episodes])
                    cells.append(values)
                    graph_rows.append(dict(**info, graph=g, **dict(zip(METRICS, values))))
                cells = np.asarray(cells)
                estimates[model] = cells
                means = finite_mean(cells)
                counts = {f"n_{m}": int(c) for m, c in zip(METRICS, np.isfinite(cells).sum(axis=0))}
                point.append(dict(**info, **dict(zip(METRICS, means)), **counts))
                boot = finite_mean(cells[draw], axis=1)
                for j, metric in enumerate(METRICS):
                    finite = boot[:, j][np.isfinite(boot[:, j])]
                    lo, hi = np.quantile(finite, [0.025, 0.975]) if finite.size else (np.nan, np.nan)
                    intervals.append(dict(**info, metric=metric, mean=means[j], low95=lo,
                                          high95=hi, n_graphs=int(np.isfinite(cells[:, j]).sum())))
            for a, b in combinations(methods, 2):
                for metric in ("ap", "recall_at_budget"):
                    j = METRICS.index(metric)
                    diff = estimates[a][:, j] - estimates[b][:, j]
                    lo, hi = np.nanquantile(finite_mean(diff[draw], axis=1), [0.025, 0.975])
                    differences.append(dict(population=pop, condition=cond, metric=metric,
                                            minuend=a, subtrahend=b, mean=finite_mean(diff),
                                            low95=lo, high95=hi))
    csv_write(out / "episode_metrics.csv", rows)
    csv_write(out / "graph_metrics.csv", graph_rows)
    csv_write(out / "summary.csv", point)
    csv_write(out / "exploratory_intervals.csv", intervals)
    csv_write(out / "exploratory_paired_differences.csv", differences)
    with (out / "classical_summary.log").open("w", encoding="utf-8") as f:
        f.write("All hosts; outbreaks averaged within graph, then equal weight across test graphs.\n")
        f.write("model                 condition                 auc       ap    recall@13  late_share\n")
        for r in point:
            if r["population"] == "all":
                f.write(f"{r['model']:21s} {r['condition']:19s} {r['auc']:9.6f} "
                        f"{r['ap']:9.6f} {r['recall_at_budget']:10.6f} {r['late_detection_share']:10.6f}\n")
