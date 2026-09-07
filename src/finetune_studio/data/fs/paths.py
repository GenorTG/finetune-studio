"""Path helpers for the per-project filesystem.

Single responsibility: figure out where things live on disk.
"""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(os.environ.get("FTS_ROOT", str(Path.home() / ".finetune-studio")))
_PROJECTS = _ROOT / "projects"


def root() -> Path:
    _ROOT.mkdir(parents=True, exist_ok=True)
    return _ROOT


def project_dir(pid: str) -> Path:
    p = _PROJECTS / pid
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_dir(pid: str, sha256: str) -> Path:
    """files/<sha256-12>/ — content-addressed."""
    short = sha256[:12]
    d = project_dir(pid) / "files" / short
    d.mkdir(parents=True, exist_ok=True)
    return d
