"""Run-manifest helpers for the Matérn Bayes-decision comparison sweeps.

A manifest is a small JSON file written alongside each sweep output file.  It
records the information needed to reproduce the run:

    git_sha     : HEAD commit hash (or "DIRTY" suffix if working tree is modified)
    config_hash : SHA-256 of the canonicalised config dict
    timestamp   : ISO-8601 UTC timestamp
    seed        : integer random seed
    hostname    : machine name for debugging

Usage::

    manifest = build_manifest(config_dict, seed)
    write_manifest(manifest, path / "manifest.json")
    manifest = read_manifest(path / "manifest.json")
"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    """Return the current HEAD SHA-1 (7 chars), or 'unknown' if git is absent."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        # Append '+dirty' if working tree has uncommitted changes
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "--exit-code"],
            stderr=subprocess.DEVNULL,
        ) != 0
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def _config_hash(cfg: dict[str, Any]) -> str:
    """Stable SHA-256 hex digest of a JSON-serialisable config dict."""
    canonical = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_manifest(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Build a manifest dict for the given config and seed."""
    return {
        "git_sha": _git_sha(),
        "config_hash": _config_hash(cfg),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "hostname": socket.gethostname(),
        "dtype": cfg["dtype"],
    }


# ---------------------------------------------------------------------------
# Persist / load
# ---------------------------------------------------------------------------

def write_manifest(manifest: dict[str, Any], path: Path | str) -> None:
    """Write manifest dict to a JSON file at ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)


def read_manifest(path: Path | str) -> dict[str, Any]:
    """Load and return a manifest dict from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
