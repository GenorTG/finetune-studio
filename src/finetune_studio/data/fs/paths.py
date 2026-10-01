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


def _safe_segment(value: str, what: str) -> str:
    """Reject values that could escape their parent directory."""
    if not value or value in {".", ".."} or any(c in value for c in "/\\\x00"):
        raise ValueError(f"invalid {what}: {value!r}")
    return value


def project_dir(pid: str) -> Path:
    p = _PROJECTS / _safe_segment(pid, "project id")
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_dir(pid: str, sha256: str, *, create: bool = True) -> Path:
    """files/<sha256-12>/ — content-addressed. Used by the data-prep runner
    for parsed files. The file library lives at files/ root, with its own
    raw/ + converted/ structure (see data.fs.file_library). Pass
    ``create=False`` for read-only lookups so they don't mkdir."""
    short = _safe_segment(sha256[:12], "sha256")
    d = project_dir(pid) / "files" / short
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def project_files_root(pid: str) -> Path:
    """files/ — root of the project's file library. Contains raw/, converted/,
    and the user-named folders under each. Created lazily by the file library
    helper on first use."""
    d = project_dir(pid) / "files"
    d.mkdir(parents=True, exist_ok=True)
    return d
