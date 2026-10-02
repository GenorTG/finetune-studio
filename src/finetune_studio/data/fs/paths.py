"""Path helpers for the per-project filesystem.

Single responsibility: figure out where things live on disk.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from fastapi import HTTPException

_ROOT = Path(os.environ.get("FTS_ROOT", str(Path.home() / ".finetune-studio")))
_PROJECTS = _ROOT / "projects"


def root() -> Path:
    _ROOT.mkdir(parents=True, exist_ok=True)
    return _ROOT


def rag_corpora_root() -> Path:
    """Root of PortableRAG corpora (``<FTS_ROOT>/rag_corpora``); never creates it."""
    return _ROOT / "rag_corpora"


def rag_corpus_dir(pid: str) -> Path:
    """Corpus directory for a project id; never creates it."""
    return rag_corpora_root() / _safe_segment(str(pid), "project id")


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


def project_files_root(pid: str, *, create: bool = True) -> Path:
    """files/ — root of the project's file library. Contains raw/, converted/,
    and the user-named folders under each. Created lazily by the file library
    helper on first use. Pass ``create=False`` for read-only existence checks
    (e.g. "does this project have any parsed files yet?") so they honour
    ``FTS_ROOT`` without creating directories."""
    d = _PROJECTS / _safe_segment(pid, "project id") / "files"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def project_roots(pid: str) -> list[Path]:
    """Resolved directories that make up project ``pid`` (never created here).

    Two trees hold project state: ``<FTS_ROOT>/projects/<pid>`` (file library,
    parsed sources) and ``<db dir>/projects/<pid>`` (datasets). Both are the
    project directory for the data-prep path fence.
    """
    from finetune_studio.db import (
        datasets as _ds,  # same ``settings`` binding as datasets_dir()
    )

    seg = _safe_segment(str(pid), "project id")
    out: list[Path] = []
    for r in (_PROJECTS / seg, Path(_ds.settings.db_path).parent / "projects" / seg):
        rr = r.resolve()
        if rr not in out:
            out.append(rr)
    return out


def resolve_within(ref: object, roots: Iterable[Path], *, what: str = "path") -> Path:
    """Resolve a user-supplied path reference and confine it to ``roots``.

    The single data-prep path fence (Genor 2026-10-02: data prep never reads or
    writes outside the project directory). Relative refs are tried against each
    root, then the cwd / ``settings.data_dir`` (legacy stored forms). The result
    is ``resolve()``d (symlinks followed) and must sit under a root, so
    absolute paths outside, ``..`` traversal and escaping symlinks all raise
    HTTP 400/403. Absolute paths that already point inside a root are fine.
    """
    raw = str(ref or "").strip()
    if not raw or "\x00" in raw:
        raise HTTPException(400, f"{what} required")
    p = Path(raw)  # no ~ expansion: "~/x" is a relative name, not $HOME
    if ".." in p.parts:
        raise HTTPException(400, f"{what}: path traversal not allowed")
    from finetune_studio.db import datasets as _ds  # db layer's ``settings`` binding

    resolved_roots = [Path(r).resolve() for r in roots]
    if p.is_absolute():
        candidates = [p]
    else:
        candidates = [r / p for r in resolved_roots]
        candidates += [Path.cwd() / p, Path(_ds.settings.data_dir) / p]
    fallback: Path | None = None
    escaped = False
    for cand in candidates:
        try:
            rc = cand.resolve()
        except (OSError, RuntimeError):  # symlink loop etc.
            continue
        if not any(rc.is_relative_to(r) for r in resolved_roots):
            escaped = escaped or rc.exists()
            continue
        if rc.exists():
            return rc
        fallback = fallback or rc
    # A ref that really names an existing file elsewhere is an escape attempt,
    # not a new in-project file (e.g. another project's ``data/projects/x/...``).
    if fallback is not None and not escaped:
        return fallback
    raise HTTPException(403, f"{what} is outside the project directory")


def resolve_in_project(pid: str, ref: object, *, what: str = "path") -> Path:
    """``resolve_within`` against project ``pid``'s directories."""
    return resolve_within(ref, project_roots(pid), what=what)
