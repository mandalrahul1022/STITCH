"""Read only graph byte literals from the September archive. Never execute pickle."""
import argparse
import json
from pathlib import Path
import pickletools
from zipfile import ZipFile

import numpy as np

from topology import edge_hash


def verified_graphs(path):
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("results/manifest.json"))
        expected = {g["hash"]: g for g in manifest["graphs"]}
        found = {}
        with archive.open("results/data.pkl") as stream:
            for op, value, _ in pickletools.genops(stream):
                if op.name == "SHORT_BINUNICODE" and value == "episodes":
                    break
                if op.name != "BINBYTES" or len(value) % 16:
                    continue
                pairs = np.frombuffer(value, dtype="<i8").reshape(-1, 2)
                if not len(pairs) or pairs.min() < 0 or pairs.max() >= 256:
                    continue
                if not np.all(pairs[:, 0] < pairs[:, 1]):
                    continue
                digest = edge_hash(pairs)
                if digest in expected:
                    found[expected[digest]["seed"]] = pairs.copy()
    splits = {"train": range(1000, 1012), "validation": range(2000, 2004),
              "test": range(3000, 3008)}
    for g in manifest["graphs"]:
        # The manifest uses 'val' on some versions.
        split = "validation" if g["split"] == "val" else g["split"]
        if g["seed"] not in splits[split]:
            raise ValueError("seed/split mismatch")
        if g["family"] != ("SBM" if g["seed"] % 2 else "BA"):
            raise ValueError("graph family mismatch")
    prescribed = {seed for seeds in splits.values() for seed in seeds}
    if set(found) != prescribed or len(expected) != 24 or len(manifest["graphs"]) != 24:
        raise ValueError("expected 24 unique, hash-verified topologies")
    return manifest["graphs"], found


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    rows, found = verified_graphs(parser.parse_args().archive)
    print(f"{len(found)} archived edge lists validated against SHA-256 and split IDs.")
