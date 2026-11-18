"""Read-only post-run provenance comparison. Neither training nor simulation imports this module."""
import argparse
import hashlib
import json
from pathlib import Path
import pickletools
from zipfile import ZipFile

import numpy as np

from stage1 import write_json


def compare(archive, data, output):
    manifest = json.loads((data / "manifest.json").read_text())
    records = {e["id"]: e for e in manifest["episodes"]}
    sizes = {57 * 256: "states", 2 * 256: "first", 256: "reference"}
    audit = []
    with ZipFile(archive) as z:
        old = {e["id"]: e for e in json.loads(z.read("results/manifest.json"))["episodes"]}
        active, fields = False, {}
        with z.open("results/data.pkl") as stream:
            for op, value, _ in pickletools.genops(stream):
                if op.name == "SHORT_BINUNICODE" and value == "episodes":
                    active = True
                if not active:
                    continue
                if op.name == "BINBYTES" and len(value) in sizes:
                    fields[sizes[len(value)]] = value
                if op.name != "SHORT_BINUNICODE" or value not in old:
                    continue
                row = records[value]
                expected = old[value]["truth_hash"]
                old_digest = hashlib.sha256(fields["states"] + fields["first"]).hexdigest()
                if old_digest != expected:
                    raise ValueError("archive provenance check failed")
                first = np.frombuffer(fields["first"], dtype="<i2")
                with np.load(data / row["file"]) as f:
                    realized = f["first"] >= 0
                    never = np.unique(first[~realized])
                    if (never <= 56).any():
                        raise ValueError("archived never-event sentinel lies inside follow-up")
                    audit.append(dict(
                        episode=value, archived_hash_verified=True,
                        archived_never_codes=never.tolist(), new_hash_differs=row["truth_hash"] != expected,
                        states_equal=np.array_equal(np.frombuffer(fields["states"], dtype=np.int8),
                                                    f["states"].ravel()),
                        realized_event_times_equal=np.array_equal(f["first"][realized], first[realized]),
                        reference_coverage_equal=np.array_equal(f["reference"],
                                                               np.frombuffer(fields["reference"], dtype=bool))))
                fields = {}
    if len(audit) != len(records) or {r["episode"] for r in audit} != set(records):
        raise ValueError("incomplete episode comparison")
    write_json(output, audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    compare(args.archive, args.data, args.out)
