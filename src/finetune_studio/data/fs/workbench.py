"""File-browser workbench backend: bulk actions, zip export, usage, search.

Companion to ``parsed_edit`` (manual parsed-text editing) — this module owns
the bulk/aggregate operations the v2 file browser exposes:

- ``bulk_action``   — delete / restore / move / reparse / tag over many ids
- ``download_zip``  — zip of raw bytes for selected files (collision-safe names)
- ``file_usage``    — where a file actually went: prep source, RAG corpus,
                       QA pairs, datasets, training runs
- ``search_content`` — substring search inside PARSED text (not just names)

All raw bytes stay immutable; nothing here deletes from disk except through
the existing soft-delete/purge paths.
"""
from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from pathlib import Path

from fastapi import HTTPException

from finetune_studio.data import project_filesystem as pfs
from finetune_studio.data.fs import file_library as fl

log = logging.getLogger(__name__)

ZIP_MAX_TOTAL_BYTES = 2 * 1024**3   # 2 GiB cap per zip request
SEARCH_MAX_FILES = 400              # parsed-text scan budget per request
SEARCH_SNIPPET_CHARS = 70


def _current_raw(pid: str, file_id: str) -> tuple[dict, Path, str]:
    f = fl.get_file(pid, file_id, include_deleted=True)
    if not f:
        raise HTTPException(status_code=404, detail=f"file {file_id} not found")
    versions = fl.list_versions(pid, file_id)
    if not versions:
        raise HTTPException(status_code=404, detail=f"file {file_id} has no version")
    match = next(
        (v for v in versions if v["version"] == f.get("current_version")), versions[0]
    )
    return f, Path(str(match.get("raw_path") or "")), str(match.get("raw_hash") or "")


# ── bulk actions ─────────────────────────────────────────────────────────


def bulk_action(pid: str, ids: list[str], action: str, payload: dict) -> dict:
    """Apply one action to many files. Returns per-id results, never partial-crashes."""
    if not ids:
        raise HTTPException(status_code=400, detail="ids required")
    if action not in ("delete", "restore", "move", "reparse", "tag-add", "tag-remove"):
        raise HTTPException(status_code=400, detail=f"unknown action {action!r}")
    results: list[dict] = []
    ok = 0
    for fid in ids[:500]:
        try:
            if action == "delete":
                fl.soft_delete_file(pid, fid)
            elif action == "restore":
                fl.restore_file(pid, fid)
            elif action == "move":
                folder_id = str(payload.get("folder_id") or "")
                if not folder_id:
                    raise HTTPException(status_code=400, detail="folder_id required")
                fl.move_file_to_folder(pid, fid, folder_id)
            elif action == "reparse":
                from finetune_studio.data.parsed_edit import reparse_file
                reparse_file(pid, fid)
            else:  # tag-add / tag-remove
                _apply_tag(pid, fid, action, payload)
            ok += 1
            results.append({"file_id": fid, "ok": True})
        except HTTPException as e:
            results.append({"file_id": fid, "ok": False, "error": str(e.detail)})
        except Exception as e:  # per-file isolation is the contract
            log.exception("bulk %s failed for %s", action, fid)
            results.append({"file_id": fid, "ok": False, "error": str(e)})
    return {"action": action, "requested": len(ids), "succeeded": ok,
            "failed": len(ids) - ok, "results": results}


def _apply_tag(pid: str, fid: str, action: str, payload: dict) -> dict:
    from finetune_studio import db

    add = _parse_tags(payload.get("tags", ""))
    if not add:
        raise HTTPException(status_code=400, detail="tags required")
    f = fl.get_file(pid, fid, include_deleted=True)
    if not f:
        raise HTTPException(status_code=404, detail="file not found")
    cur = _parse_tags(f.get("tags") or "")
    if action == "tag-add":
        merged = list(dict.fromkeys(cur + add))
    else:
        drop = {t.lower() for t in add}
        merged = [t for t in cur if t.lower() not in drop]
    with db.cursor() as c:
        c.execute("UPDATE project_files SET tags = ? WHERE id = ? AND project_id = ?",
                  (", ".join(merged), fid, pid))
    return {"ok": True, "tags": ", ".join(merged)}


def _parse_tags(raw) -> list[str]:
    if isinstance(raw, list):
        items = raw
    else:
        items = re.split(r"[,;]", str(raw or ""))
    return [s.strip() for s in items if str(s).strip()]


# ── zip export ───────────────────────────────────────────────────────────


def download_zip(pid: str, ids: list[str]) -> tuple[bytes, str]:
    """Zip the raw bytes of the selected files. Returns (payload, filename)."""
    if not ids:
        raise HTTPException(status_code=400, detail="ids required")
    from finetune_studio import db

    proj = db.get_project(pid) or {}
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(proj.get("name") or pid)).strip("-") or pid
    buf = io.BytesIO()
    seen: set[str] = set()
    added = 0
    total = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fid in ids[:500]:
            try:
                f, raw, _hash = _current_raw(pid, fid)
            except HTTPException:
                continue
            if not raw.is_file():
                continue
            size = raw.stat().st_size
            if total + size > ZIP_MAX_TOTAL_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"selection exceeds the {ZIP_MAX_TOTAL_BYTES // (1024**3)} GiB zip cap",
                )
            total += size
            arc = _unique_arcname(f.get("original_name") or fid, seen)
            zf.write(raw, arcname=arc)
            added += 1
    if added == 0:
        raise HTTPException(status_code=404, detail="no readable files for those ids")
    return buf.getvalue(), f"{slug}-files-{added}.zip"


def _unique_arcname(name: str, seen: set[str]) -> str:
    arc = re.sub(r"[/\\]+", "_", name).strip("_") or "file"
    if arc in seen:
        stem, _, ext = arc.rpartition(".")
        n = 2
        while f"{stem}-{n}.{ext}" in seen:
            n += 1
        arc = f"{stem}-{n}.{ext}"
    seen.add(arc)
    return arc


# ── per-file usage ───────────────────────────────────────────────────────


def file_usage(pid: str, file_id: str) -> dict:
    """Where this file's content actually went: source → pairs → datasets → runs,
    plus RAG corpus membership."""
    f, raw, raw_hash = _current_raw(pid, file_id)
    from finetune_studio.data.parsed_edit import _source_for_file

    source = _source_for_file(pid, raw, raw_hash)
    out: dict = {
        "file_id": file_id,
        "name": f.get("original_name"),
        "has_override": bool(raw) and (raw.parent / (raw.name + ".parsed.md")).is_file(),
        "source": None,
        "qa_pairs": 0,
        "rag": False,
        "datasets": [],
        "runs": [],
    }
    if not source:
        return out
    src_id = str(source.get("id") or "")
    out["source"] = {
        "id": src_id,
        "status": source.get("status"),
        "parser": source.get("parser"),
        "chunk_count": int(source.get("chunk_count") or 0),
        "char_count": int(source.get("char_count") or 0),
    }
    from finetune_studio.data.parsed_edit import corpus_sha12s

    out["rag"] = src_id in corpus_sha12s(pid)

    pair_ids = {p.get("id") for p in pfs.list_qa_pairs(pid, source_id=src_id) if p.get("id")}
    out["qa_pairs"] = len(pair_ids)
    if not pair_ids:
        return out

    from finetune_studio import db

    ds_used: list[dict] = []
    ds_paths: set[str] = set()
    for ds in db.list_datasets(pid):
        dp = str(ds.get("data_path") or "")
        if not dp or not Path(dp).is_file():
            continue
        if _jsonl_contains_ids(dp, pair_ids):
            ds_used.append({"id": ds.get("id"), "name": ds.get("name"),
                            "data_path": dp, "qa_count": ds.get("qa_count")})
            ds_paths.add(str(Path(dp).resolve()))
    out["datasets"] = ds_used

    for run in db.list_runs(project_id=pid):
        rp = str(run.get("data_path") or "")
        if rp and str(Path(rp).resolve()) in ds_paths:
            out["runs"].append({"id": run.get("id"), "name": run.get("name"),
                                "status": run.get("status"),
                                "base_model": run.get("base_model")})
    return out


def _jsonl_contains_ids(path: str, pair_ids: set[str]) -> bool:
    """True when any line's id (or provenance source_id) is one of pair_ids.
    Bounded scan — datasets are small by design; cap keeps worst case sane."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i > 20000:
                    return False
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                rid = str(row.get("id") or row.get("qa_id") or "")
                if rid and rid in pair_ids:
                    return True
    except OSError:
        return False
    return False


# ── parsed-content search ────────────────────────────────────────────────


def search_content(pid: str, q: str, limit: int = 25) -> dict:
    """Substring search inside parsed text of every file (name-search stays
    client-side). Returns snippets around the first hit per file."""
    q = (q or "").strip()
    if len(q) < 2:
        raise HTTPException(status_code=400, detail="query needs >= 2 chars")
    ql = q.lower()
    matches: list[dict] = []
    scanned = 0
    skipped = 0
    for f in fl.list_files(pid):
        if scanned >= SEARCH_MAX_FILES or len(matches) >= limit:
            break
        scanned += 1
        try:
            doc = fl.get_parsed_markdown(pid, str(f["id"]))
        except HTTPException:
            skipped += 1
            continue
        except Exception:  # noqa: BLE001  # unreadable parse for one file ≠ 500
            skipped += 1
            continue
        text = str(doc.get("parsed_md") or "")
        pos = text.lower().find(ql)
        if pos < 0:
            continue
        start = max(0, pos - SEARCH_SNIPPET_CHARS)
        end = min(len(text), pos + len(q) + SEARCH_SNIPPET_CHARS)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()
        matches.append({
            "file_id": str(f["id"]),
            "name": f.get("original_name"),
            "parsed_source": doc.get("source"),
            "occurrences": text.lower().count(ql),
            "snippet": ("…" if start > 0 else "") + snippet + ("…" if end < len(text) else ""),
        })
    return {"query": q, "matches": matches, "scanned": scanned,
            "skipped_unparseable": skipped,
            "truncated": scanned >= SEARCH_MAX_FILES or len(matches) >= limit}
