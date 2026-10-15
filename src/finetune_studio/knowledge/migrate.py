"""One-time move from the old two-layer file storage to ``knowledge.sources`` (2026-10-14).

Before: every parsed file had a JSON manifest under ``qa/sources/<sha12>.json``, a byte copy of the upload plus a
``metadata.json`` under ``files/<sha12>/``, and a hand edit lived in a ``<raw>.parsed.md`` file next to the upload.
After: the parse state is the ``file_conversions`` row of the library file; ``files/<sha12>/`` holds only
``parsed.txt``, ``parsed.json`` and ``chunks/``.

Idempotent: a project without ``qa/sources/`` has nothing to move. Runs at service start (``webui.app``),
from ``update.sh`` and after a project import. Nothing reads the old layout afterwards.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from finetune_studio import db
from finetune_studio.data.fs.paths import file_dir, project_dir
from finetune_studio.knowledge import sources

log = logging.getLogger(__name__)

_KEEP = {"parsed.txt", "parsed.json", "chunks"}


def needs_migration(pid: str) -> bool:
    return (project_dir(pid) / "qa" / "sources").is_dir()


def migrate_project(pid: str) -> dict[str, int]:
    """Move one project's source manifests into the DB and drop the old files. Returns counts for the log."""
    root = project_dir(pid)
    manifests_dir = root / "qa" / "sources"
    stats = {"sources": 0, "ready": 0, "queued": 0, "orphans": 0, "edits": 0}
    if manifests_dir.is_dir():
        for p in sorted(manifests_dir.glob("*.json")):
            try:
                manifest = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("migrate %s: unreadable manifest %s dropped", pid, p.name)
                continue
            stats[_migrate_source(pid, manifest)] += 1
            stats["sources"] += 1
        shutil.rmtree(manifests_dir)
    for r in db.file_parses.rows(pid, include_deleted=True):
        stats["edits"] += _fold_override(pid, r)
        _drop_stale_copies(pid, str(r.get("raw_hash") or ""))
    log_file = root / "logs" / "ingestions.jsonl"
    log_file.unlink(missing_ok=True)
    (root / "project.json").unlink(missing_ok=True)
    return stats


def migrate_brain_storage() -> dict[str, int]:
    """One-time move of the RAG-era storage (2026-10-14): ``$FTS_ROOT/rag_corpora`` -> ``$FTS_ROOT/brain`` (index
    rows re-pointed), per-project ``rag-bundles`` -> ``brain-bundles`` and the export settings keys. Idempotent."""
    from finetune_studio.data.fs.paths import brain_root
    from finetune_studio.smart_brain.export_config import migrate_settings_keys

    out = {"indexes": 0, "rows": 0, "bundles": 0, "settings": 0}
    old = brain_root().parent / "rag_corpora"
    new = brain_root()
    if old.is_dir():
        if new.exists():
            for d in old.iterdir():
                if not (new / d.name).exists():
                    d.rename(new / d.name)
                    out["indexes"] += 1
            if not any(old.iterdir()):
                old.rmdir()
        else:
            out["indexes"] = sum(1 for d in old.iterdir() if d.is_dir())
            old.rename(new)
    out["rows"] = db.brain_indexes.rebase_store_paths(str(old), str(new))
    for project in db.list_projects():
        pdir = project_dir(str(project["id"]))
        if (pdir / "rag-bundles").is_dir() and not (pdir / "brain-bundles").exists():
            (pdir / "rag-bundles").rename(pdir / "brain-bundles")
            out["bundles"] += 1
    out["settings"] = int(migrate_settings_keys())
    return out


def migrate_all() -> int:
    """Migrate every project still on the old layout (and the RAG-era storage); returns how many projects moved."""
    moved = migrate_brain_storage()
    if any(moved.values()):
        log.info("Smart Brain storage migration: %s", moved)
    n = 0
    for project in db.list_projects():
        pid = str(project["id"])
        if needs_migration(pid):
            log.info("knowledge migration %s: %s", pid, migrate_project(pid))
            n += 1
    return n


def _migrate_source(pid: str, manifest: dict) -> str:
    sid = str(manifest.get("id") or "")[:12]
    row = db.file_parses.row_for_hash(pid, sid, include_deleted=True) if sid else None
    if row is None and manifest.get("file_id"):
        row = db.file_parses.row_for_file(pid, str(manifest["file_id"]))
    if row is None:                       # a data-prep upload that never entered the file library: add it now
        row = _adopt(pid, manifest)
    if row is None:
        log.warning("migrate %s: source %s (%s) has no bytes left anywhere; its pairs keep their id",
                    pid, sid, manifest.get("filename"))
        return "orphans"
    if row.get("status"):                 # already has a parse row (a second manifest for the same bytes)
        return "ready" if row["status"] == sources.READY else "queued"
    d = file_dir(pid, str(row["raw_hash"]), create=False)
    parsed = d / "parsed.txt"
    chunks = sources.passages(pid, str(row["raw_hash"]))
    if manifest.get("status") == "ready" and parsed.is_file() and chunks:
        text = parsed.read_text(encoding="utf-8", errors="replace")
        db.file_parses.upsert(
            str(row["file_id"]), int(row["version"]), status=sources.READY, converter=str(manifest.get("parser") or ""),
            converted_path=str(parsed), converted_size=len(text.encode("utf-8")), char_count=len(text),
            chunk_count=len(chunks), edited_at=manifest.get("edited_at"),
        )
        return "ready"
    db.file_parses.upsert(str(row["file_id"]), int(row["version"]), status=sources.QUEUED,
                          converted_path=str(parsed))
    return "queued"


def _adopt(pid: str, manifest: dict) -> dict | None:
    """Put the bytes of a source that bypassed the file library into it (same bytes = same source id)."""
    from finetune_studio.data.fs import file_library as fl

    raw = Path(str(manifest.get("data_path") or manifest.get("path") or ""))
    if not raw.is_file():
        return None
    meta = fl.write_uploaded_file(pid, raw.read_bytes(), str(manifest.get("filename") or raw.name),
                                  mime_hint=str(manifest.get("mime_type") or "") or None, uploaded_by="migration")
    return db.file_parses.row_for_file(pid, meta.file_id)


def _fold_override(pid: str, row: dict) -> int:
    """A ``<raw>.parsed.md`` hand edit becomes the source text (if it differs), then the override file goes."""
    raw = Path(str(row.get("raw_path") or ""))
    override = raw.parent / (raw.name + ".parsed.md")
    if not raw.name or not override.is_file():
        return 0
    text = override.read_text(encoding="utf-8", errors="replace")
    folded = 0
    source = sources.source_for_file(pid, str(row["file_id"]))
    if source is not None and source.ready and text.strip() and text != sources.parsed_text(pid, source):
        sources.set_text(pid, source.id, text)
        folded = 1
    override.unlink()
    return folded


def _drop_stale_copies(pid: str, raw_hash: str) -> None:
    """``files/<sha12>/`` keeps only the parse; the old byte copy and ``metadata.json`` go."""
    if not raw_hash:
        return
    d = file_dir(pid, raw_hash, create=False)
    if not d.is_dir():
        return
    for p in d.iterdir():
        if p.name in _KEEP:
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
