"""C05 Step A: compare a Stage 2 run against September's classical_summary.log.

Read-only. Prints markdown tables and writes them to <run>/c05_compare.md.
Usage:
  python tools/compare_september.py --run <stage2 out> --data <stage1 data> --zip <Feasibility zip>
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path

METRICS = ("auc", "ap", "recall_at_budget", "late_detection_share")
TOL = {"auc": 0.03, "ap": 0.02}


def september(zip_path):
    z = zipfile.ZipFile(zip_path)
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


def ours(run, population):
    with (run / "summary.csv").open() as f:
        return {(r["model"], r["condition"]): {m: float(r[m]) for m in METRICS}
                for r in csv.DictReader(f) if r["population"] == population}


def table(sep, new, population):
    out = [f"### Population: {population}", "",
           "| model | condition | " + " | ".join(f"{m} new / Sep / Δ" for m in METRICS) + " | flag |",
           "|---|---|" + "---|" * (len(METRICS) + 1)]
    flags, missing = [], []
    for key in sorted(set(sep) | set(new)):
        if key not in new or key not in sep:
            missing.append(f"{key}: {'missing in new run' if key not in new else 'not in September'}")
            continue
        cells, flag = [], []
        for m in METRICS:
            d = new[key][m] - sep[key][m]
            cells.append(f"{new[key][m]:.4f} / {sep[key][m]:.4f} / {d:+.4f}")
            if m in TOL and abs(d) > TOL[m]:
                flag.append(f"{m} {d:+.4f}")
        if flag:
            flags.append((key, flag))
        out.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + f" | {'; '.join(flag)} |")
    out += ["", f"Flagged cells (|ΔAP|>0.02 or |ΔAUC|>0.03): {len(flags)}"]
    out += [f"- {k[0]} / {k[1]}: {', '.join(f)}" for k, f in flags]
    if missing:
        out += ["", "Unmatched rows:"] + [f"- {m}" for m in missing]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--zip", type=Path, required=True)
    a = p.parse_args()

    lines = ["## C05 Step A", "", "### Verification"]
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

    sep = september(a.zip)
    for pop in ("all", "candidate"):
        lines += [""] + table(sep, ours(a.run, pop), pop)

    req = [(m, c) for m in ("fitted_mechanistic", "boosted_trees") for c in ("full", "bridge_persistent")]
    have = ours(a.run, "all")
    lines += ["", "Required rows present: " + ", ".join(f"{m}/{c}={'yes' if (m, c) in have else 'NO'}"
                                                       for m, c in req)]
    text = "\n".join(lines) + "\n"
    (a.run / "c05_compare.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
