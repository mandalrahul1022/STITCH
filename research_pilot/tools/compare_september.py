"""C05 Step A: compare Stage 2 runs against September's classical_summary.log.

Read-only. Prints markdown tables and writes them to <run>/c05_compare.md.
Usage:
  python tools/compare_september.py --run <stage2 out> --data <stage1 data> --zip <Feasibility zip>
         [--old <earlier stage2 out>]
With --old, each cell shows September / old / new, and the flag uses new - September.
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np

METRICS = ("auc", "ap", "recall_at_budget", "late_detection_share")
TOL = {"auc": 0.03, "ap": 0.02}


def september(z):
    name = next(n for n in z.namelist() if n.endswith("classical_summary.log"))
    rows = {}
    for line in z.read(name).decode().splitlines():
        tok = line.split()
        if len(tok) != 6:
            continue
        try:
            vals = [float(v) for v in tok[2:]]
        except ValueError:
            continue
        rows[(tok[0], tok[1])] = dict(zip(METRICS, vals))
    return rows


def summary(run, population):
    with (run / "summary.csv").open() as f:
        return {(r["model"], r["condition"]): {k: float(v) for k, v in r.items()
                                               if k not in ("population", "condition", "model") and v != ""}
                for r in csv.DictReader(f) if r["population"] == population}


def table(sep, new, old, population, flag):
    head = "Sep / old / new / Δ(new-Sep)" if old else "new / Sep / Δ"
    out = [f"### Population: {population}", "",
           "| model | condition | " + " | ".join(f"{m} {head}" for m in METRICS)
           + " | recall per-origin (new) | flag |",
           "|---|---|" + "---|" * (len(METRICS) + 2)]
    flags = []
    for key in sorted(new):
        if key not in sep:
            continue
        cells, hit = [], []
        for m in METRICS:
            d = new[key][m] - sep[key][m]
            if old and key in old:
                cells.append(f"{sep[key][m]:.4f} / {old[key][m]:.4f} / {new[key][m]:.4f} / {d:+.4f}")
            else:
                cells.append(f"{new[key][m]:.4f} / {sep[key][m]:.4f} / {d:+.4f}")
            if flag and m in TOL and abs(d) > TOL[m]:
                hit.append(f"{m} {d:+.4f}")
        if hit:
            flags.append((key, hit))
        po = new[key].get("recall_at_budget_per_origin", float("nan"))
        out.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + f" | {po:.4f} | {'; '.join(hit)} |")
    if flag:
        out += ["", f"Flagged cells (|ΔAP|>0.02 or |ΔAUC|>0.03 vs September): {len(flags)}"]
        out += [f"- {k[0]} / {k[1]}: {', '.join(f)}" for k, f in flags]
    return out


def zero_origin_counts(run):
    with (run / "episode_metrics.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if "origins_zero_positive" in r]
    if not rows:
        return ["", "origins_zero_positive not recorded in this run."]
    tot = {}
    for r in rows:
        k = (r["population"], r["condition"])
        tot.setdefault(k, {})[r["episode"]] = int(r["origins_zero_positive"])
    lines = ["", "Primary origins excluded from per-origin recall (zero positives), summed over test episodes:", "",
             "| population | condition | excluded origins | of |", "|---|---|---|---|"]
    for (pop, cond), s in sorted(tot.items()):
        lines.append(f"| {pop} | {cond} | {sum(s.values())} | {16 * len(s)} |")
    return lines


def hidden_set(path):
    with np.load(path) as f:
        return set(np.flatnonzero(f["x"][12, :, 4] == 0).tolist())


def mask_overlap(z, data):
    sep = json.loads(z.read(next(n for n in z.namelist() if n.endswith("mask_overlap.json"))))
    manifest = json.loads((data / "manifest.json").read_text())
    files = {(v["episode"], v["condition"]): data / v["file"] for v in manifest["views"]}
    keys = sorted(sep[0])
    lines = ["", "### Mask overlap", "",
             f"September's mask_overlap.json has keys {keys}: the share of the bridge set also in the "
             "degree-matched control, per episode. It stores no host IDs, so a direct "
             "ours-vs-September comparison of bridge or region sets is not possible from it.", "",
             "| episode | family | Sep bridge∩matched | ours bridge∩matched | ours bridge∩region |",
             "|---|---|---|---|---|"]
    for r in sep:
        ep = r["episode"]
        try:
            b, m, g = (hidden_set(files[(ep, c)]) for c in
                       ("bridge_persistent", "degree_matched", "region_persistent"))
            ours = f"{len(b & m) / len(b):.4f} | {len(b & g) / len(b):.4f}"
        except KeyError:
            ours = "n/a | n/a"
        lines.append(f"| {ep} | {r.get('family', '')} | {r['fraction_bridge_set_shared']:.4f} | {ours} |")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--old", type=Path)
    a = p.parse_args()
    z = zipfile.ZipFile(a.zip)

    lines = ["## C05 Step A" + (" (C05b rerun)" if a.old else ""), "", "### Verification"]
    s1 = json.loads((a.data / "verification.json").read_text())["tests"]
    lines += [f"- Stage 1 {t['id']}: {t['status']}" for t in s1]
    v2 = a.run / "verification.json"
    if v2.exists():
        lines += [f"- Stage 2 {k}: {v}" for k, v in json.loads(v2.read_text()).items()]
    else:
        lines.append("- Stage 2 verification.json MISSING (run verify_classical.py --out)")

    prev = json.loads((a.run / "bridge_prevalence.json").read_text())
    lines += ["", "### bridge_persistent candidate prevalence (September 0.0758)", "",
              "| rule | graph_macro | pooled | positives | eligible |", "|---|---|---|---|---|"]
    lines += [f"| {r} | {d['graph_macro']:.4f} | {d['pooled']:.4f} | {d['positives']} | {d['eligible']} |"
              for r, d in prev.items()]

    sep = september(z)
    for pop in ("all", "candidate"):
        old = summary(a.old, pop) if a.old else None
        lines += [""] + table(sep, summary(a.run, pop), old, pop, flag=(pop == "all"))
    lines += zero_origin_counts(a.run)
    lines += mask_overlap(z, a.data)

    req = [(m, c) for m in ("fitted_mechanistic", "boosted_trees") for c in ("full", "bridge_persistent")]
    have = summary(a.run, "all")
    lines += ["", "Required rows present: " + ", ".join(f"{m}/{c}={'yes' if (m, c) in have else 'NO'}"
                                                       for m, c in req)]
    text = "\n".join(lines) + "\n"
    (a.run / "c05_compare.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
