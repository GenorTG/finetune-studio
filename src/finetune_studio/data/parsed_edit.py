"""Manual parsed-text editing + pipeline status (file workbench, Genor 2026-09-20).

One concern: a human edits the PARSED text of a library file (fixing OCR
garbage, trimming boilerplate) and that edit must flow into BOTH consumers —
the data-prep/Q&A chunks (training data) and the RAG corpus (on next build) —
while the original raw bytes stay immutable.

Storage model (mirrors ``file_library.get_parsed_markdown`` resolution order):
  - The manual override lives at ``<raw>.md`` (sibling of the stored raw
    file) — resolution step 3, so it wins over on-the-fly conversion.
  - When the file is already a data-prep QA source, the canonical parse
    artifact ``files/<sha12>/parsed.txt`` is rewritten too and the chunks are
    regenerated, so prep jobs and RAG builds see the edited text.

``reparse`` discards the override and re-runs the real parser from raw bytes.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from fastapi import HTTPException

from finetune_studio.data import project_filesystem as pfs
from finetune_studio.data.fs import file_library as fl

log = logging.getLogger(__name__)

MANUAL_PARSER = "manual-edit"


def _current_raw_path(pid: str, file_id: str) -> tuple[dict, Path, str]:
    f = fl.get_file(pid, file_id)
    if not f:
        raise HTTPException(status_code=404, detail="file not found")
    versions = fl.list_versions(pid, file_id)
    if not versions:
        raise HTTPException(status_code=404, detail="file has no stored version")
    match = next(
        (v for v in versions if v["version"] == f.get("current_version")), versions[0]
    )
    raw = Path(str(match.get("raw_path") or ""))
    if not raw.is_file():
        raise HTTPException(status_code=410, detail="raw bytes missing on disk")
    return f, raw, str(match.get("raw_hash") or "")


def _source_for_file(pid: str, raw: Path, raw_hash: str = "") -> dict | None:
    """Find the QA source for a library file — by content sha256 first (the
    source may have been registered from the content-addressed copy, not the
    raw/ path), then by path as a fallback."""
    for s in pfs.list_qa_sources(pid):
        if raw_hash and str(s.get("sha256") or "") == raw_hash:
            return s
    try:
        want = str(raw.resolve())
    except OSError:
        want = str(raw)
    for s in pfs.list_qa_sources(pid):
        for key in ("data_path", "path"):
            val = s.get(key) or ""
            if val and str(Path(val)) == want:
                return s
    return None


def _rewrite_source_chunks(pid: str, source: dict, text: str) -> dict:
    """Point a QA source at hand-edited text: parsed.txt + chunks + manifest."""
    sha = str(source.get("sha256") or "")
    if not sha:
        return source
    fd = pfs.file_dir(pid, sha)
    (fd / "parsed.txt").write_text(text, encoding="utf-8")
    from finetune_studio.data.prep.chunker import chunk_text

    chunks = chunk_text(text)
    pfs.write_chunks(pid, sha, chunks)
    try:
        pfs.update_file_metadata(pid, sha, chunk_count=len(chunks), char_count=len(text))
    except Exception:  # metadata.json may not exist for old sources
        log.debug("metadata update skipped for %s", sha[:12], exc_info=True)
    updated = {
        **source,
        "char_count": len(text),
        "chunk_count": len(chunks),
        "parser": MANUAL_PARSER,
        "status": "ready",
        "edited_at": time.time(),
    }
    pfs.write_qa_source(pid, updated)
    return updated


def _override_path(raw: Path) -> Path:
    """Manual-edit override file — never the raw file itself (a ``.md``
    upload would collide with ``raw.with_suffix('.md')`` and destroy the
    original bytes).``<raw>.parsed.md`` is collision-proof."""
    return raw.parent / (raw.name + ".parsed.md")


def save_parsed_override(pid: str, file_id: str, text: str) -> dict:
    """Save a human-edited parsed text for a library file.

    Writes the ``<raw>.parsed.md`` override (invalidating the parsed cache),
    and — when the file is a data-prep source — rewrites ``parsed.txt`` and
    regenerates chunks so training data + the next RAG build use the edit.
    """
    if text is None:
        raise HTTPException(status_code=400, detail="text required")
    f, raw, raw_hash = _current_raw_path(pid, file_id)
    _override_path(raw).write_text(text, encoding="utf-8")
    fl.invalidate_parsed_cache(pid, file_id)

    source = _source_for_file(pid, raw, raw_hash)
    rechunked = False
    source_id = None
    chunks = 0
    if source is not None:
        source = _rewrite_source_chunks(pid, source, text)
        rechunked = True
        source_id = source.get("id")
        chunks = int(source.get("chunk_count") or 0)
    return {
        "ok": True,
        "file_id": file_id,
        "name": f.get("original_name"),
        "chars": len(text),
        "source_id": source_id,
        "rechunked": rechunked,
        "chunk_count": chunks,
        "rag_note": (
            "RAG corpus rebuild picks up the edit" if rechunked
            else "send to Data Prep (⚡) to feed training + RAG"
        ),
    }


def reparse_file(pid: str, file_id: str) -> dict:
    """Discard the manual override and re-run the real parser from raw bytes."""
    f, raw, raw_hash = _current_raw_path(pid, file_id)
    override = _override_path(raw)
    removed = False
    if override.is_file():
        override.unlink()
        removed = True
    fl.invalidate_parsed_cache(pid, file_id)

    source = _source_for_file(pid, raw, raw_hash)
    out: dict = {"ok": True, "file_id": file_id, "override_removed": removed,
                 "name": f.get("original_name")}
    if source is not None:
        from finetune_studio.data.prep.ingest import parse_and_chunk

        data = raw.read_bytes()
        sha = str(source.get("sha256") or "")
        ingest = parse_and_chunk(
            pid, data, str(source.get("filename") or f.get("original_name") or raw.name),
            sha256=sha, mime_type=str(source.get("mime_type") or f.get("mime_type") or ""),
            reuse_if_parsed=False,
        )
        if ingest.ok:
            updated = {
                **source,
                "char_count": ingest.char_count,
                "chunk_count": ingest.chunk_count,
                "parser": ingest.parser,
                "status": "ready",
            }
            updated.pop("edited_at", None)
            pfs.write_qa_source(pid, updated)
            out.update({"source_id": source.get("id"),
                        "chunk_count": ingest.chunk_count,
                        "parser": ingest.parser})
        else:
            out["error"] = ingest.error or "reparse failed"
    return out


_RAG_SHA12_RE = re.compile(r"/files/([0-9a-f]{12})/")


def _corpus_sha12s(pid: str) -> set[str]:
    """sha12 dirs indexed in the project's RAG corpus, from the manifest.

    Corpus ``document_id`` is an md5 of the source path — NOT the content
    sha12 — so the badge must read ``documents_meta[].source`` paths
    (``…/files/<sha12>/parsed.txt``) instead. Best-effort: no manifest →
    empty set. Checks both the rag.py location (~) and FTS_ROOT for tests.
    """
    from finetune_studio.data.fs.paths import root

    out: set[str] = set()
    for base in (Path.home() / ".finetune-studio" / "rag_corpora",
                 root() / "rag_corpora"):
        manifest = base / pid / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            meta = (raw.get("extra") or {}).get("documents_meta") or []
            for d in meta:
                m = _RAG_SHA12_RE.search(str(d.get("source") or ""))
                if m:
                    out.add(m.group(1))
        except (OSError, ValueError):
            log.debug("rag manifest unreadable for %s", pid, exc_info=True)
        break
    return out


def pipeline_status(pid: str) -> dict[str, dict]:
    """Per-file pipeline flags for the browser: parsed / prep / rag.

    ``in_rag`` matches the QA source id (content sha12) against the corpus
    manifest's indexed ``files/<sha12>/`` paths. No corpus → all False.
    """
    status: dict[str, dict] = {}
    sources = pfs.list_qa_sources(pid)
    by_sha: dict[str, dict] = {}
    by_path: dict[str, dict] = {}
    for s in sources:
        sha = str(s.get("sha256") or "")
        if sha:
            by_sha[sha] = s
        for key in ("data_path", "path"):
            val = s.get(key)
            if val:
                by_path[str(Path(val))] = s

    rag_ids = _corpus_sha12s(pid)

    for f in fl.list_files(pid):
        fid = str(f["id"])
        try:
            versions = fl.list_versions(pid, fid)
        except Exception:  # noqa: BLE001
            versions = []
        raw_path = ""
        raw_hash = ""
        for v in versions:
            if v.get("version") == f.get("current_version"):
                raw_path = str(v.get("raw_path") or "")
                raw_hash = str(v.get("raw_hash") or "")
                break
        if not raw_path and versions:
            raw_path = str(versions[0].get("raw_path") or "")
            raw_hash = str(versions[0].get("raw_hash") or "")
        src = (by_sha.get(raw_hash) if raw_hash else None) or by_path.get(raw_path)
        src_ready = bool(src) and (
            str(src.get("status") or "ready") == "ready" or int(src.get("chunk_count") or 0) > 0
        )
        rp = Path(raw_path) if raw_path else None
        sibling = rp.with_suffix(".md") if rp else None
        has_parsed = bool(rp) and (
            _override_path(rp).is_file()
            or (sibling is not None and sibling != rp and sibling.is_file())
            or src_ready
        )
        source_id = src.get("id") if src else None
        in_rag = bool(source_id) and str(source_id) in rag_ids
        status[fid] = {
            "has_parsed": has_parsed,
            "source_id": source_id,
            "chunk_count": int((src or {}).get("chunk_count") or 0),
            "parser": (src or {}).get("parser") or "",
            "in_rag": in_rag,
        }
    return status
