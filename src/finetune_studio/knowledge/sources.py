"""Sources: the one parse of every project file (Files & Knowledge, Genor 2026-10-14).

A source IS a file-library file (``project_files`` + its current ``file_versions`` row). Its parse state is the
``file_conversions`` row with ``format = 'parsed'`` (``db.file_parses``); the parsed text and its passages live on disk:

    files/<sha12>/parsed.txt     the text every consumer reads (parser output, hand edits, Vision readings)
    files/<sha12>/parsed.json    the parser's full result (metadata, warnings)
    files/<sha12>/chunks/NNNN.txt the passages: mining keys items and pairs by their 1-based index

``source_id`` is the first 12 hex of the raw bytes' sha256: mining items, pairs and the folder above use it. Training
(mining -> pairs -> dataset) and the Smart Brain index read the same text and passages; nothing parses a file twice.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finetune_studio import db
from finetune_studio.data.fs.paths import file_dir
from finetune_studio.knowledge.errors import SourceError

log = logging.getLogger(__name__)

QUEUED, PARSING, READY, ERROR = "queued", "parsing", "ready", "error"
MANUAL_EDIT = "manual-edit"
_MIN_TEXT = 10          # a parse shorter than this holds no facts: it is an error, not an empty source


@dataclass(frozen=True)
class Source:
    id: str                 # sha12 of the current raw bytes
    file_id: str
    version: int
    sha256: str
    filename: str
    mime_type: str
    raw_path: str
    size_bytes: int
    uploaded_at: float
    status: str             # queued | parsing | ready | error
    text_sha: str           # sha256 of parsed.txt ('' until parsed): what the index records it was built from
    parser: str
    char_count: int
    chunk_count: int
    error: str
    edited_at: float | None
    deleted: bool

    @property
    def ready(self) -> bool:
        return self.status == READY and self.chunk_count > 0

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ready": self.ready}


def _from_row(r: dict[str, Any]) -> Source:
    raw_hash = str(r.get("raw_hash") or "")
    return Source(
        id=raw_hash[:12], file_id=str(r["file_id"]), version=int(r["version"]), sha256=raw_hash,
        filename=str(r.get("original_name") or ""), mime_type=str(r.get("mime_type") or ""),
        raw_path=str(r.get("raw_path") or ""), size_bytes=int(r.get("size_bytes") or 0),
        uploaded_at=float(r.get("uploaded_at") or 0.0), status=str(r.get("status") or QUEUED),
        text_sha=str(r.get("converted_hash") or ""),
        parser=str(r.get("converter") or ""), char_count=int(r.get("char_count") or 0),
        chunk_count=int(r.get("chunk_count") or 0), error=str(r.get("error_message") or ""),
        edited_at=r.get("edited_at"), deleted=r.get("deleted_at") is not None,
    )


def is_parseable(filename: str) -> bool:
    """True when a parser (or the LibreOffice render fallback) can read ``filename``."""
    from finetune_studio.data.parsers import readable_extensions

    return Path(filename or "").suffix.lower() in readable_extensions()


# ── lookups ───────────────────────────────────────────────────────────────────────────────────────────────────────

def list_sources(pid: str) -> list[Source]:
    """Every live file that has been staged for parsing, newest upload first."""
    out = [_from_row(r) for r in db.file_parses.rows(pid) if r.get("status")]
    return sorted(out, key=lambda s: s.uploaded_at, reverse=True)


def get_source(pid: str, source_id: str, *, include_deleted: bool = False) -> Source | None:
    r = db.file_parses.row_for_hash(pid, str(source_id or "")[:12], include_deleted=include_deleted)
    return _from_row(r) if r and r.get("status") else None


def require_source(pid: str, source_id: str) -> Source:
    s = get_source(pid, source_id)
    if s is None:
        raise SourceError(f"unknown source {source_id!r} in project {pid}")
    return s


def source_for_file(pid: str, file_id: str) -> Source | None:
    r = db.file_parses.row_for_file(pid, file_id)
    return _from_row(r) if r and r.get("status") else None


def names(pid: str) -> dict[str, str]:
    """``source_id -> filename`` for labels (pairs, datasets, coverage)."""
    return {s.id: s.filename for s in list_sources(pid)}


def source_dir(pid: str, source: Source, *, create: bool = True) -> Path:
    return file_dir(pid, source.sha256, create=create)


def parsed_text(pid: str, source: Source) -> str:
    p = source_dir(pid, source, create=False) / "parsed.txt"
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""


def require_file_source(pid: str, file_id: str) -> Source:
    """The source of a library file, or a ``SourceError`` saying why there is none."""
    r = db.file_parses.row_for_file(pid, file_id)
    if r is None:
        raise SourceError(f"file {file_id} not found")
    if not r.get("status"):
        raise SourceError(f"{r.get('original_name')}: no parser reads this file type", 422)
    return _from_row(r)


def view(pid: str, file_id: str) -> dict[str, Any]:
    """The parsed text of a library file for the editor/preview. Not parsed yet = 409, parse failed = 422."""
    s = require_file_source(pid, file_id)
    if s.status in (QUEUED, PARSING):
        raise SourceError(f"{s.filename} is still being parsed", 409)
    if s.status == ERROR:
        raise SourceError(f"{s.filename}: {s.error or 'the parser failed'}", 422)
    return {"fid": file_id, "source_id": s.id, "parser": s.parser, "edited_at": s.edited_at,
            "chunk_count": s.chunk_count, "parsed_md": parsed_text(pid, s)}


def passages(pid: str, sha256: str) -> list[str]:
    """The source's passages in index order (index ``i`` = mining chunk ``i + 1``).

    Indexed by each file's own ``NNNN`` name: a missing file stays an empty gap instead of shifting every later
    passage down one (mining items and pair provenance key on the index).
    """
    chunks_dir = file_dir(pid, sha256, create=False) / "chunks"
    if not chunks_dir.is_dir():
        return []
    out: list[str] = []
    for p in sorted(chunks_dir.glob("*.txt")):
        try:
            idx = int(p.stem)
        except ValueError:
            continue
        while len(out) < idx:
            log.warning("passage %04d.txt missing under %s: kept as a gap", len(out), chunks_dir)
            out.append("")
        try:
            out.append(p.read_text(encoding="utf-8"))
        except OSError:
            out.append("")
    return out


# ── state changes ─────────────────────────────────────────────────────────────────────────────────────────────────

def stage(pid: str, file_id: str) -> Source:
    """Mark a library file for parsing (cheap: the upload request calls it, the parse runs after the response).

    A file already parsed (or being parsed) is returned unchanged; a failed or queued one is queued again.
    """
    r = db.file_parses.row_for_file(pid, file_id)
    if r is None:
        raise SourceError(f"file {file_id} not found in project {pid}")
    if r.get("status") in (READY, PARSING):
        return _from_row(r)
    if not is_parseable(str(r.get("original_name") or "")):
        raise SourceError(f"{r.get('original_name')}: no parser reads this file type", 422)
    db.file_parses.upsert(file_id, int(r["version"]), status=QUEUED, error_message=None,
                          converted_path=str(file_dir(pid, str(r["raw_hash"]), create=False) / "parsed.txt"))
    return _reload(pid, file_id)


def parse(pid: str, source_id: str, *, force: bool = False) -> Source:
    """Parse the source's raw bytes into ``parsed.txt`` + passages. Idempotent unless ``force``.

    ``force`` re-runs the parser and discards a hand edit or Vision reading (Reparse). The returned source is
    ``ready`` or ``error`` (with the reason); it never raises for a bad file.
    """
    s = require_source(pid, source_id)
    if not force and s.ready and (source_dir(pid, s, create=False) / "parsed.txt").is_file():
        return s
    raw = Path(s.raw_path)
    if not raw.is_file():
        return _fail(pid, s, f"missing file: {s.raw_path}")
    db.file_parses.upsert(s.file_id, s.version, status=PARSING, error_message=None)
    from finetune_studio.data.parsers import parse_bytes

    try:
        result = parse_bytes(s.filename, raw.read_bytes())
    except Exception as e:  # a parser crash is this file's error, reported on the source
        log.exception("parse failed for %s", s.filename)
        return _fail(pid, s, f"parser crashed: {e}")
    text = str(result.get("text") or "")
    meta = result.get("metadata") or {}
    if len(text.strip()) < _MIN_TEXT:
        why = "; ".join(str(w) for w in (meta.get("warnings") or []) if w)
        return _fail(pid, s, "the parser found no text" + (f" ({why})" if why else ""),
                     parser=str(meta.get("parser") or ""))
    d = source_dir(pid, s)
    (d / "parsed.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return _write_text(pid, s, text, parser=str(meta.get("parser") or ""), edited=False)


def ensure_parsed(pid: str, source_id: str) -> Source:
    return parse(pid, source_id, force=False)


def set_text(pid: str, source_id: str, text: str, *, parser: str = MANUAL_EDIT) -> Source:
    """Replace the parsed text (a hand edit, or figures read by the Vision reader) and rebuild the passages."""
    if len((text or "").strip()) < _MIN_TEXT:
        raise SourceError("the text is empty: write at least a sentence, or Reparse the file", 400)
    return _write_text(pid, require_source(pid, source_id), text, parser=parser, edited=True)


def forget_hashes(pid: str, raw_hashes: list[str]) -> None:
    """Drop everything derived from these raw versions: pairs, mining records and ``files/<sha12>/``.

    The file library calls it when a file is purged (after its own rows are gone); a trashed (soft-deleted) file
    keeps its knowledge until then.
    """
    from finetune_studio.data.fs import qa as qa_fs

    for raw_hash in dict.fromkeys(h for h in raw_hashes if h):
        sid = raw_hash[:12]
        qa_fs.delete_pairs_of_source(pid, sid)
        db.mining.delete_file_records(pid, sid)
        d = file_dir(pid, raw_hash, create=False)
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def resume_unfinished() -> int:
    """Parse every source a restart left ``queued`` or ``parsing`` (the upload's background parse died with the
    process). Runs in a startup thread; returns how many sources it parsed."""
    n = 0
    for project in db.list_projects():
        pid = str(project["id"])
        for s in list_sources(pid):
            if s.status in (QUEUED, PARSING):
                parse(pid, s.id, force=True)
                n += 1
    return n


# ── internals ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _reload(pid: str, file_id: str) -> Source:
    s = source_for_file(pid, file_id)
    if s is None:  # pragma: no cover - the row was written a moment ago
        raise SourceError(f"file {file_id} vanished while staging")
    return s


def _fail(pid: str, s: Source, reason: str, *, parser: str = "") -> Source:
    db.file_parses.upsert(s.file_id, s.version, status=ERROR, error_message=reason, chunk_count=0,
                          **({"converter": parser} if parser else {}))
    return _reload(pid, s.file_id)


def _write_text(pid: str, s: Source, text: str, *, parser: str, edited: bool) -> Source:
    from finetune_studio.data.fs.chunks import write_chunks
    from finetune_studio.data.prep.chunker import chunk_text

    d = source_dir(pid, s)
    parsed = d / "parsed.txt"
    parsed.write_text(text, encoding="utf-8")
    chunks = chunk_text(text)
    if not chunks:
        return _fail(pid, s, "the text split into no passages", parser=parser)
    write_chunks(pid, s.sha256, chunks)
    data = text.encode("utf-8")
    db.file_parses.upsert(
        s.file_id, s.version, status=READY, error_message=None, converter=parser, converted_path=str(parsed),
        converted_hash=hashlib.sha256(data).hexdigest(), converted_size=len(data), char_count=len(text),
        chunk_count=len(chunks), edited_at=time.time() if edited else None,
    )
    return _reload(pid, s.file_id)
