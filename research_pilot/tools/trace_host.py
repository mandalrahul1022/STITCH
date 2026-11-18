"""C04 host trace: follow one host from truth to observation channels, label, candidacy and a score.

Run from the repository root:
    python -m research_pilot.tools.trace_host --data research_pilot/results/stage1/run_a \
        --episode 3000_1 --host 17 --origins 11,13,15,17,19
    python -m research_pilot.tools.trace_host --fixture t08

The true state, queue flag and reference membership are printed for audit only. They are NOT
model inputs. Channels are printed compactly per mask condition as "<visible state> a<age> k<seen>":
  visible state = S/E/I/P from channels 0-3 when channel 4 = 1, "-" when hidden (channels 0-4 all 0)
  a<age>        = channel 5 x 32 (ticks since last observation, clipped at 32)
  k<seen>       = channel 6 (observer has ever seen this host as E, I or P)
Channel 7 (normalized log-degree) is constant per host and printed once. Every compact cell is
decoded back to the 8 stored floats and compared, so the printout is lossless.
The comparator score is fitted_mechanistic, fitted here on the training views of the full condition
exactly as stage2.py does, with observer-known hosts forced to 0.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PILOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PILOT))

from classical import Dynamics, adjacency, predict  # noqa: E402
from pilot import E, HISTORY, I, P, S, Process, advance, features, labels, visible_evidence  # noqa: E402
from stage1 import FAMILIAR, HELD_OUT  # noqa: E402

LETTERS = "SEIP"


def compact(x_t):
    """8 channels of one host at one tick -> 'I a00 k1'; verified lossless against x_t[:7]."""
    onehot, avail, age, seen = x_t[:4], x_t[4], x_t[5], x_t[6]
    letter = LETTERS[int(np.argmax(onehot))] if avail == 1 else "-"
    text = f"{letter} a{int(round(age * 32)):02d} k{int(seen)}"
    decoded = np.zeros(7, dtype=np.float32)
    if letter != "-":
        decoded[LETTERS.index(letter)] = 1
        decoded[4] = 1
    decoded[5], decoded[6] = int(text[3:5]) / 32, int(text[-1])
    if not np.array_equal(decoded, x_t[:7]):
        raise AssertionError(f"compact encoding is lossy: {x_t[:7]} -> {text}")
    return text


def checks(first_v, views, host, ticks):
    """The three C04 accept criteria, evaluated on the stored arrays."""
    out = []
    c = int(first_v)
    if c > 0:
        y = np.array([labels(np.array([c]), t)[0] for t in range(ticks)])
        on = np.flatnonzero(y)
        ok = on.size and on[0] == max(c - 5, 0) and on[-1] == c - 1 and on.size == min(c, 5)
        out.append(("label is 1 exactly for C-5 <= t <= C-1", bool(ok),
                    f"C={c}: y=1 for t={on[0]}..{on[-1]}, 0 at t={c - 6 if c >= 6 else 'n/a'} and t={c}"))
    else:
        out.append(("label never 1 for a seed host or a never-compromised host",
                    not any(labels(np.array([c]), t)[0] for t in range(ticks)), f"C={c}"))
    for cond, x in views.items():
        h = x[:, host]
        hidden = h[:, 4] == 0
        zeros = not h[hidden, :4].any()
        out.append((f"{cond}: hidden ticks have channels 0-4 all 0", bool(zeros),
                    f"{int(hidden.sum())} hidden ticks"))
        seen = h[:, 6]
        onset = np.flatnonzero(np.diff(np.r_[0, seen]) > 0)
        valid = all(h[t, 4] == 1 and np.argmax(h[t, :4]) in (E, I, P) and h[t, :4].any() for t in onset)
        out.append((f"{cond}: channel 6 turns on only at a visible E/I/P tick", bool(valid),
                    f"onset tick {onset.tolist() or 'never'}"))
    return out


def print_ticks(states, queues, views, host, ticks):
    conds = list(views)
    print("| tick | true | queue (NOT AN INPUT) | " + " | ".join(conds) + " |")
    print("|---" * (3 + len(conds)) + "|")
    for t in ticks:
        cells = [compact(views[c][t, host]) for c in conds]
        print(f"| {t} | {LETTERS[states[t, host]]} | {int(queues[t, host])} | " + " | ".join(cells) + " |")


def print_checks(results):
    print("\n| check | result | detail |\n|---|---|---|")
    for name, ok, detail in results:
        print(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    return all(ok for _, ok, _ in results)


def episode_trace(data, episode_id, host, origins):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    ep = next(e for e in manifest["episodes"] if e["id"] == episode_id)
    graph = next(g for g in manifest["graphs"] if g["seed"] == ep["graph"])
    with np.load(data / ep["file"]) as f:
        states, first, queues, reference = f["states"], f["first"], f["queues"], f["reference"]
    with np.load(data / f"graph_{ep['graph']}.npz") as f:
        g = {k: f[k] for k in f.files}
    adj = adjacency(g["edges"])
    views = {}
    for v in manifest["views"]:
        if v["episode"] == episode_id:
            with np.load(data / v["file"]) as f:
                views[v["condition"]] = f["x"]
    views = {c: views[c] for c in FAMILIAR + HELD_OUT if c in views}
    train = []
    for v in manifest["views"]:
        e = next(e for e in manifest["episodes"] if e["id"] == v["episode"])
        if e["split"] == "train" and v["condition"] == "full":
            with np.load(data / v["file"]) as f, np.load(data / f"graph_{e['graph']}.npz") as gf:
                train.append((f["x"], adjacency(gf["edges"])))
    model = Dynamics.fit(train)
    bad = [t for t in origins if t < HISTORY - 1]
    if bad:
        raise SystemExit(f"origins need {HISTORY} ticks of history: {bad}")
    scores = {c: predict("fitted_mechanistic", model, x, origins, adj, None)[:, host]
              for c, x in views.items()}

    degree = int(np.bincount(g["edges"].ravel(), minlength=256)[host])
    sets = [name for name in ("bridge_persistent", "degree_matched", "region_persistent")
            if name in g and host in g[name]]
    print(f"### Episode {episode_id}, host {host}\n")
    print(f"- split {ep['split']}, graph {ep['graph']} ({graph['family']}), beta {ep['beta']}, "
          f"outbreak seed {ep['seed']}")
    print(f"- degree {degree}; channel 7 (normalized log-degree) = {views['full'][0, host, 7]:.4f}")
    print(f"- C(v) = {int(first[host])}; reference-observed (NOT AN INPUT): {bool(reference[host])}")
    print(f"- member of held-out host sets: {', '.join(sets) or 'none'}")
    print(f"- fitted_mechanistic (fitted on {len(train)} training full views): hazard {model.hazard:.5f}, "
          f"progress {model.progress:.5f}, remove {model.remove:.5f}\n")
    print_ticks(states, queues, views, host, range(len(states)))
    print("\n| origin t | y(v,t) | " + " | ".join(f"{c}: status, score" for c in views) + " |")
    print("|---" * (2 + len(views)) + "|")
    for j, t in enumerate(origins):
        y = int(labels(first, t)[host])
        cells = [f"{'known' if views[c][t, host, 6] else 'candidate'}, {scores[c][j]:.4f}" for c in views]
        print(f"| {t} | {y} | " + " | ".join(cells) + " |")
    return print_checks(checks(first[host], views, host, len(states)))


def fixture_trace():
    """T08 tiny graph: host 0 infectious from tick 0, host 1 susceptible; the event fires at tick 4."""
    ticks = 11
    state = np.array([I, S], dtype=np.uint8)
    first = np.array([0, -1])
    queued = np.zeros(2, dtype=bool)
    reference = np.array([False, True])
    edges = np.array([[0, 1]])
    states, queues = [state.copy()], [queued.copy()]
    quiet, forced = Process(0, 0, 1, 1, 1), Process(1, 0, 1, 1, 1)
    rng = np.random.default_rng(4)
    for tick in range(1, ticks):
        advance(state, first, queued, reference, edges, tick, forced if tick >= 4 else quiet, rng)
        states.append(state.copy())
        queues.append(queued.copy())
    states, queues = np.array(states), np.array(queues)
    masks = {"full": np.ones_like(states, dtype=bool)}
    hidden = np.ones_like(states, dtype=bool)
    hidden[3:6, 1] = False
    masks["host1_hidden_t3_5"] = hidden
    views = {c: features(visible_evidence(states, m), np.ones(2), (E, I, P)) for c, m in masks.items()}
    print("### T08 fixture, host 1\n")
    print("- 2 hosts, one edge; host 0 is I from tick 0 and not reference-observed; host 1 is "
          "reference-observed. Ticks 1-3 use beta 0; from tick 4, beta 1, volatility 0, progression 1, "
          "enqueue fraction 1, completion 1.")
    print(f"- C(v) = {int(first[1])}; channel 7 = 1.0 (both hosts have degree 1)")
    print("- Second view hides host 1 at ticks 3-5 to show channel 6 waiting for a visible tick.")
    print("- No comparator score: the fixture has no fitted model.\n")
    print_ticks(states, queues, views, 1, range(ticks))
    print("\n| origin t | y(v,t) | " + " | ".join(f"{c}: status" for c in views) + " |")
    print("|---" * (2 + len(views)) + "|")
    for t in range(0, 6):
        cells = ["known" if views[c][t, 1, 6] else "candidate" for c in views]
        print(f"| {t} | {int(labels(first, t)[1])} | " + " | ".join(cells) + " |")
    return print_checks(checks(first[1], views, 1, ticks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--episode")
    parser.add_argument("--host", type=int)
    parser.add_argument("--origins", default="13,15,17")
    parser.add_argument("--fixture", choices=("t08",))
    args = parser.parse_args()
    if args.fixture:
        ok = fixture_trace()
    else:
        if args.data is None or args.episode is None or args.host is None:
            parser.error("--data, --episode and --host are required without --fixture")
        ok = episode_trace(args.data, args.episode, args.host,
                           [int(t) for t in args.origins.split(",")])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
