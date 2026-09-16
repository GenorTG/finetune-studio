"""Human-readable labels for RAG sources / citations.

Project files live under ``files/<sha12>/`` with opaque deliverable names
(``parsed.txt``, ``chunks/0000.txt``). Citations must show the original
upload name so documents remain distinguishable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_OPAQUE_BASENAMES = frozenset({
    "parsed.txt",
    "parsed.json",
    "metadata.json",
})
_CHUNK_FILE_RE = re.compile(r"^\d{4}\.txt$")


def is_opaque_source_name(name: str) -> bool:
    """True when *name* is a pipeline artifact, not a user-facing filename."""
    base = Path(name or "").name
    if not base:
        return True
    if base in _OPAQUE_BASENAMES:
        return True
    return bool(_CHUNK_FILE_RE.match(base))


def _read_original_filename(file_dir: Path) -> str | None:
    """Return ``original_filename`` from ``metadata.json`` if present."""
    meta_path = file_dir / "metadata.json"
    if not meta_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    for key in ("original_filename", "original_name", "filename"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    aliases = data.get("aliases")
    if isinstance(aliases, list):
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                return alias.strip()
    return None


def _fallback_original_in_dir(file_dir: Path) -> str | None:
    """Pick a non-opaque sibling file name when metadata is missing."""
    try:
        for child in sorted(file_dir.iterdir()):
            if not child.is_file():
                continue
            if child.name in _OPAQUE_BASENAMES:
                continue
            if child.suffix.lower() in {".json", ".jsonl"}:
                continue
            return child.name
    except OSError:
        return None
    return None


def original_name_for_file_dir(file_dir: Path) -> str:
    """Best-effort original upload name for a content-addressed file directory."""
    found = _read_original_filename(file_dir)
    if found:
        return found
    found = _fallback_original_in_dir(file_dir)
    if found:
        return found
    return file_dir.name


def display_name_for_path(path: str | Path) -> str:
    """Resolve a filesystem path to a citation-friendly display name."""
    p = Path(path)
    name = p.name

    if p.parent.name == "chunks" and _CHUNK_FILE_RE.match(name):
        base = original_name_for_file_dir(p.parent.parent)
        try:
            idx = int(p.stem)
        except ValueError:
            idx = p.stem
        return f"{base} (chunk {idx})"

    if name in {"parsed.txt", "parsed.json"}:
        base = original_name_for_file_dir(p.parent)
        kind = "parsed" if name.startswith("parsed") else name
        if base and base != name and base != p.parent.name:
            return f"{base} ({kind})"
        return f"{p.parent.name} ({kind})"

    if is_opaque_source_name(name):
        # Generic opaque artifact under a sha dir
        if (p.parent / "metadata.json").is_file() or (p.parent / "parsed.txt").is_file():
            base = original_name_for_file_dir(p.parent)
            return f"{base} ({name})"
        return f"source ({name})"

    return name


def prettify_source_label(
    filename: str | None,
    source: str | Path | None = None,
) -> str:
    """Pretty-print a stored filename, optionally using the source path."""
    fname = (filename or "").strip()
    # Prefer an already-human filename stored on the chunk row.
    if fname and not is_opaque_source_name(fname):
        return fname
    if source:
        pretty = display_name_for_path(source)
        src_name = Path(source).name
        if pretty and pretty != src_name:
            return pretty
        if pretty and not is_opaque_source_name(pretty):
            return pretty
    if source and (not fname or is_opaque_source_name(fname)):
        return display_name_for_path(source)
    if not fname:
        return "unknown source"
    if fname in {"parsed.txt", "parsed.json"}:
        return f"parsed source ({fname})"
    if _CHUNK_FILE_RE.match(fname):
        return f"chunk ({fname})"
    return fname


_RAW_FILE_ID_RE = re.compile(r"^([0-9a-f]{12,64})_(.+)$", re.IGNORECASE)


def _files_root_for_raw_path(path: Path) -> Path | None:
    """Return ``…/files`` when *path* lives under ``files/raw/…``."""
    for parent in (path, *path.parents):
        if parent.name == "raw" and parent.parent.name == "files":
            return parent.parent
    return None


def _is_file_library_raw(path: Path) -> bool:
    """True for immutable file-library uploads under ``files/raw/``."""
    return _files_root_for_raw_path(path) is not None


def _has_canonical_parsed_for_raw(path: Path) -> bool:
    """True when a content-addressed ``files/<sha12>/parsed.txt`` exists.

    Raw library names are ``{file_id}_{stem}.ext`` where ``file_id`` is the
    content sha prefix; auto-promote writes ``files/<sha12>/parsed.txt``.
    """
    files_root = _files_root_for_raw_path(path)
    if files_root is None:
        return False
    match = _RAW_FILE_ID_RE.match(path.name)
    if not match:
        return False
    short = match.group(1)[:12].lower()
    return (files_root / short / "parsed.txt").is_file()


def should_ingest_source_file(path: Path) -> bool:
    """Whether *path* should be indexed into a project RAG corpus.

    Skips ``chunks/NNNN.txt`` shards. In content-addressed file dirs (those
    with ``metadata.json`` or ``parsed.txt``), only ``parsed.txt`` is ingested
    so uploads and chunk shards are not double-indexed under opaque names.

    Also skips file-library ``files/raw/…`` copies when a canonical parsed
    source for the same content hash already exists — rebuild must index one
    logical document, not raw+parsed twins. Standalone text outside
    ``files/raw/`` (or raw uploads with no parsed twin) still ingest.
    """
    if not path.is_file():
        return False
    if path.parent.name == "chunks":
        return False
    if _CHUNK_FILE_RE.match(path.name):
        return False
    if ".RAW_TRASH" in path.parts or ".CONVERTED_TRASH" in path.parts:
        return False
    parent = path.parent
    in_store = (parent / "metadata.json").is_file() or (parent / "parsed.txt").is_file()
    if in_store:
        return path.name == "parsed.txt"
    # Skip file-library raw twins when a canonical parsed source exists.
    return not (
        _is_file_library_raw(path) and _has_canonical_parsed_for_raw(path)
    )
