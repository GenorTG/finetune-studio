"""Shared parse + chunk steps for data-prep ingest.

Used by DataPrepRunner (upload/start) and the promote-from-library path
so both write ``files/<sha12>/parsed.txt`` where ``read_source`` looks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finetune_studio.data import project_filesystem as pfs
from finetune_studio.data.fs.paths import file_dir
from finetune_studio.data.prep.chunker import chunk_text


@dataclass
class IngestResult:
    """Outcome of parse + chunk for one content-addressed file."""

    ok: bool
    sha256: str
    text: str = ""
    chunks: list[str] = field(default_factory=list)
    parser: str = ""
    parser_version: str = ""
    char_count: int = 0
    chunk_count: int = 0
    warnings: list[Any] = field(default_factory=list)
    error: str = ""
    reused: bool = False


def _parsed_path(pid: str, sha256: str) -> Path:
    return file_dir(pid, sha256) / "parsed.txt"


def is_already_parsed(pid: str, sha256: str) -> bool:
    """True when parsed.txt exists, is non-trivial, and is not a placeholder.

    Parser error strings ("[DOC: ... install antiword]", "[Failed ...",
    "[Unsupported ...]") write >10 chars, so the old length check served
    them forever and a failed parse could never be retried after the
    parser dependency was installed (QABUG 2026-09-18, .doc/olefile).
    """
    p = _parsed_path(pid, sha256)
    if not p.is_file():
        return False
    try:
        text = p.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return False
    if len(text) < 10:
        return False
    return not text.startswith(("[DOC:", "[Failed", "[Unsupported", "[EPUB:"))


def load_existing_chunks(pid: str, sha256: str) -> list[str]:
    """Load chunk texts previously written under files/<sha>/chunks/."""
    chunks_dir = file_dir(pid, sha256) / "chunks"
    if not chunks_dir.is_dir():
        return []
    out: list[str] = []
    for p in sorted(chunks_dir.glob("*.txt")):
        try:
            out.append(p.read_text(encoding="utf-8"))
        except OSError:
            continue
    return out


def parse_and_chunk(
    pid: str,
    data: bytes,
    filename: str,
    *,
    sha256: str,
    max_chunks: int = 0,
    mime_type: str = "",
    reuse_if_parsed: bool = True,
) -> IngestResult:
    """Parse bytes, write parsed outputs + chunks, update file metadata.

    Steps match DataPrepRunner stages 2–5. When ``reuse_if_parsed`` and
    ``parsed.txt`` already exists, skip re-parsing and return existing
    text/chunks.
    """
    if reuse_if_parsed and is_already_parsed(pid, sha256):
        text = _parsed_path(pid, sha256).read_text(encoding="utf-8", errors="replace")
        chunks = load_existing_chunks(pid, sha256)
        if not chunks:
            chunks = chunk_text(text)
            if max_chunks and len(chunks) > max_chunks:
                chunks = chunks[:max_chunks]
            if chunks:
                pfs.write_chunks(pid, sha256, chunks)
                pfs.update_file_metadata(pid, sha256, chunk_count=len(chunks))
        meta = pfs.read_file_metadata(pid, sha256)
        return IngestResult(
            ok=True,
            sha256=sha256,
            text=text,
            chunks=chunks,
            parser=(meta.parser if meta else "") or "",
            parser_version=(meta.parser_version if meta else "") or "",
            char_count=len(text),
            chunk_count=len(chunks),
            warnings=list(meta.warnings) if meta and meta.warnings else [],
            reused=True,
        )

    from finetune_studio.data.parsers import parse_bytes

    result = parse_bytes(filename, data)
    text = result.get("text", "") or ""
    result_meta = result.get("metadata", {}) or {}
    if not text or len(text.strip()) < 10:
        return IngestResult(
            ok=False,
            sha256=sha256,
            parser=str(result_meta.get("parser", "") or ""),
            error="empty parse",
            warnings=list(result_meta.get("warnings", []) or []),
        )

    pfs.write_parsed_outputs(pid, sha256, text, result)
    pfs.update_file_metadata(
        pid,
        sha256,
        mime_type=mime_type or "",
        char_count=len(text),
        parser=result_meta.get("parser", ""),
        parser_version=result_meta.get("version", ""),
        warnings=result_meta.get("warnings", []),
    )

    chunks = chunk_text(text)
    if max_chunks and len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
    if not chunks:
        return IngestResult(
            ok=False,
            sha256=sha256,
            text=text,
            parser=str(result_meta.get("parser", "") or ""),
            char_count=len(text),
            error="no chunks",
            warnings=list(result_meta.get("warnings", []) or []),
        )

    pfs.write_chunks(pid, sha256, chunks)
    pfs.update_file_metadata(pid, sha256, chunk_count=len(chunks))
    return IngestResult(
        ok=True,
        sha256=sha256,
        text=text,
        chunks=chunks,
        parser=str(result_meta.get("parser", "") or ""),
        parser_version=str(result_meta.get("version", "") or ""),
        char_count=len(text),
        chunk_count=len(chunks),
        warnings=list(result_meta.get("warnings", []) or []),
        reused=False,
    )


def ensure_qa_source_parsed(pid: str, source: dict) -> dict:
    """Parse + chunk a registered QA source if not already ready.

    Reads bytes from ``data_path``/``path``, stores content-addressed if
    needed, runs :func:`parse_and_chunk`, and updates the QA source
    manifest to ``status=ready`` with chunk/parser/char counts.
    """
    path_str = source.get("data_path") or source.get("path") or ""
    p = Path(path_str)
    if not p.is_file():
        source = {**source, "status": "error", "error": f"missing file: {path_str}"}
        pfs.write_qa_source(pid, source)
        return source

    data = p.read_bytes()
    filename = source.get("filename") or source.get("name") or p.name
    mime = source.get("mime_type") or ""

    # Ensure content-addressed store exists (parsed outputs live under file_dir).
    _, meta = pfs.store_file(
        pid,
        data,
        original_filename=filename,
        mime_type=mime,
        source_kind="promote",
    )
    sha256 = meta.sha256
    ingest = parse_and_chunk(
        pid,
        data,
        filename,
        sha256=sha256,
        mime_type=mime,
        reuse_if_parsed=True,
    )
    if not ingest.ok:
        source = {
            **source,
            "id": source.get("id") or sha256[:12],
            "sha256": sha256,
            "status": "error",
            "error": ingest.error or "parse failed",
            "chunk_count": 0,
            "parser": ingest.parser,
            "char_count": ingest.char_count,
        }
        pfs.write_qa_source(pid, source)
        return source

    source = {
        **source,
        "id": source.get("id") or sha256[:12],
        "sha256": sha256,
        "filename": filename,
        "name": source.get("name") or filename,
        "mime_type": mime,
        "char_count": ingest.char_count,
        "chunk_count": ingest.chunk_count,
        "parser": ingest.parser,
        "status": "ready",
    }
    source.pop("error", None)
    pfs.write_qa_source(pid, source)
    return source
