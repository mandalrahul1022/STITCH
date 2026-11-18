"""C00 environment check: record the runtime this machine offers, without producing results.

Run from the repository root:
    python -m research_pilot.tools.env_check [--out PATH] [--device cuda|cpu]
"""
import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "research_pilot" / "environment_local.json"
PACKAGES = ("numpy", "scipy", "sklearn", "networkx", "torch_geometric",
            "joblib", "threadpoolctl")
BANNED = ("torch_scatter", "torch_sparse", "torch_cluster")


def version(name):
    try:
        return getattr(importlib.import_module(name), "__version__", "unknown")
    except ImportError:
        return None


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect(requested):
    import torch

    cuda = torch.cuda.is_available()
    device = requested or ("cuda" if cuda else "cpu")
    if device == "cuda" and not cuda:
        raise SystemExit("--device cuda requested but CUDA is not available")
    info = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "os": platform.platform(),
        "machine": platform.node(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "compute_capability": list(torch.cuda.get_device_capability(0)) if device == "cuda" else None,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": {name: version(name) for name in PACKAGES},
        "banned_packages_importable": [name for name in BANNED if version(name) is not None],
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_tracked_files": (git("status", "--porcelain", "--untracked-files=no") or "").splitlines(),
    }
    info["git_dirty"] = bool(info["git_dirty_tracked_files"])
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    args = parser.parse_args()
    info = collect(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"wrote {args.out}")
    if info["banned_packages_importable"]:
        raise SystemExit(f"banned packages importable: {info['banned_packages_importable']}")


if __name__ == "__main__":
    main()
