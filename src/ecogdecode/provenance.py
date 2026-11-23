"""Provenance: the environment and code version behind every result, and a log of every look at the test labels.

Every script that reads the official test labels calls :func:`record_test_look`, which appends one line to
``results/test_looks.jsonl`` with the time, the purpose, the git commit and whether the working tree had
uncommitted changes. Committing before a run therefore makes each look auditable (METHODS.md, section 10).
"""

from __future__ import annotations

import datetime
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
import torch


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_state() -> dict:
    """The current commit and whether tracked files differ from it."""
    return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain", "--untracked-files=no"))}


def environment(threads: int | None = None) -> dict:
    """Versions, platform and code state, stored with every results file."""
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        **git_state(),
    }
    if threads is not None:
        env["threads"] = threads
    return env


def record_test_look(purpose: str, detail: dict | None = None, path: str | Path = "results/test_looks.jsonl") -> None:
    """Append one look at the official test labels to the log."""
    entry = {
        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "purpose": purpose,
        **git_state(),
        **(detail or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
