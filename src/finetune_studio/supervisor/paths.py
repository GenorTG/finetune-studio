"""Where the supervisor keeps its runtime files (derived from FTS_ROOT, no extra knob)."""
from __future__ import annotations

import os
from pathlib import Path


def run_dir() -> Path:
    """``<FTS_ROOT>/run`` (default ``~/.finetune-studio/run``); not created here."""
    root = os.environ.get("FTS_ROOT") or str(Path.home() / ".finetune-studio")
    return Path(root) / "run"


def socket_path() -> Path:
    return run_dir() / "supervisor.sock"


def lock_path() -> Path:
    return run_dir() / "supervisor.lock"


def events_path() -> Path:
    return run_dir() / "events.jsonl"
