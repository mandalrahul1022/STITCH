"""C01 code hash: SHA-256 per source file under research_pilot/ plus one combined hash.

Run from the repository root:
    python -m research_pilot.tools.code_hash [--check]

Rules
- Every file under research_pilot/ is hashed, except: the output folders results/, data/ and
  classical_results/, __pycache__/, files under
  tests/ larger than 1 MB, code_manifest.json itself (self-reference) and environment*.json
  (machine-specific runtime records that would change the hash on every machine).
- Text files are hashed with CRLF converted to LF, so a Windows checkout (core.autocrlf=true)
  and a Linux checkout give the same hash. Binary files are hashed as raw bytes.
- The combined hash is SHA-256 over the sorted lines "<posix path>\t<sha256>\n".
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

PILOT = Path(__file__).resolve().parents[1]
MANIFEST = PILOT / "code_manifest.json"
BASE_COMMIT = "43ca102ee8de944acf2177cb915db945999d08f9"
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".log", ".yaml", ".yml",
                 ".toml", ".cfg", ".ini"}
FIXTURE_LIMIT = 1024 * 1024
# Run outputs, not source. data/ and classical_results/ are the inherited Stage 1-2 output
# folders; they are pinned by git and by data/manifest.json hashes instead.
OUTPUT_DIRS = ("results", "data", "classical_results")


def excluded(rel: Path, size: int) -> bool:
    parts = rel.parts
    if parts[0] in OUTPUT_DIRS or "__pycache__" in parts or rel.suffix in (".pyc", ".pyo"):
        return True
    if parts[0] == "tests" and size > FIXTURE_LIMIT:
        return True
    return rel.as_posix() == "code_manifest.json" or (len(parts) == 1 and rel.name.startswith("environment")
                                                       and rel.suffix == ".json")


def file_hash(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def compute(root: Path = PILOT) -> dict:
    files = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if not excluded(rel, path.stat().st_size):
            files[rel.as_posix()] = file_hash(path)
    lines = "".join(f"{name}\t{digest}\n" for name, digest in sorted(files.items()))
    return {
        "base_commit": BASE_COMMIT,
        "combined_sha256": hashlib.sha256(lines.encode("utf-8")).hexdigest(),
        "n_files": len(files),
        "normalization": "CRLF->LF for text suffixes " + ",".join(sorted(TEXT_SUFFIXES)),
        "excluded": ["results/", "data/", "classical_results/", "__pycache__/", "tests/ files > 1 MB", "code_manifest.json",
                     "environment*.json (top level)"],
        "sha256": files,
    }


def code_hash(root: Path = PILOT) -> str:
    """Combined hash of the current tree; what run directories record."""
    return compute(root)["combined_sha256"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="compare with the saved code_manifest.json instead of writing it")
    args = parser.parse_args()
    current = compute()
    if args.check:
        saved = json.loads(MANIFEST.read_text(encoding="utf-8"))
        changed = sorted(set(saved.get("sha256", {}).items()) ^ set(current["sha256"].items()))
        names = sorted({name for name, _ in changed})
        print(f"saved {saved.get('combined_sha256')}\ncurrent {current['combined_sha256']}")
        for name in names:
            print(f"  differs: {name}")
        sys.exit(0 if saved.get("combined_sha256") == current["combined_sha256"] else 1)
    MANIFEST.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    print(f"{current['n_files']} files hashed")
    print(f"combined code hash: {current['combined_sha256']}")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
