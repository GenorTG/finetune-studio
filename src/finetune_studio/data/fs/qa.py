"""Q&A pair storage on disk: ``qa/pairs/<qa-id>.json`` per pair (per project).

A pair's ``source_id`` names the file it was mined from (``knowledge.sources``).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from finetune_studio.data.fs.paths import project_dir

log = logging.getLogger(__name__)

# Parsed pair files keyed by directory, then file name -> (mtime_ns, size, record). A listing stats every file (cheap) and
# re-reads only the ones that changed, so reviewing thousands of pairs does not re-parse every JSON file per click.
# Validation is by stat, not by invalidation hooks: pairs written by the miner, the Guide or another process are picked up too.
_PAIR_CACHE: dict[str, dict[str, tuple[int, int, dict]]] = {}
_PAIR_CACHE_LOCK = threading.Lock()


def write_qa_pair(pid: str, qa: dict) -> None:
    qa_dir = project_dir(pid) / "qa" / "pairs"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / f"{qa['id']}.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    _forget_cached_pair(qa_dir, f"{qa['id']}.json")


def _forget_cached_pair(pairs_dir: Path, name: str) -> None:
    """Drop one file from the listing cache. A rewrite within the same filesystem timestamp tick can keep both
    mtime and size (``pending`` -> ``approved`` plus a shorter ``updated_at``), so stat alone would serve the stale pair."""
    with _PAIR_CACHE_LOCK:
        _PAIR_CACHE.get(str(pairs_dir), {}).pop(name, None)


def list_qa_pairs(pid: str, source_id: str | None = None, status: str | None = None) -> list[dict]:
    pairs_dir = project_dir(pid) / "qa" / "pairs"
    if not pairs_dir.exists():
        return []
    key = str(pairs_dir)
    with _PAIR_CACHE_LOCK:
        known = _PAIR_CACHE.get(key, {})
    fresh: dict[str, tuple[int, int, dict]] = {}
    skipped = 0
    with os.scandir(pairs_dir) as it:
        entries = sorted((e for e in it if e.name.endswith(".json")), key=lambda e: e.name)
    for entry in entries:
        try:
            st = entry.stat()
            hit = known.get(entry.name)
            if hit is not None and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
                fresh[entry.name] = hit
                continue
            fresh[entry.name] = (st.st_mtime_ns, st.st_size, json.loads(Path(entry.path).read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            skipped += 1
    with _PAIR_CACHE_LOCK:
        _PAIR_CACHE[key] = fresh
    out = []
    for _mtime, _size, qa in fresh.values():
        if source_id and qa.get("source_id") != source_id:
            continue
        if status and qa.get("status") != status:
            continue
        out.append(dict(qa))  # callers may edit what they get; never hand out the cached object itself
    if skipped:
        log.warning("list_qa_pairs(%s): skipped %d corrupt/unreadable pair file(s) under %s", pid, skipped, pairs_dir)
    return out


def update_qa_pair(pid: str, qa_id: str, **fields) -> dict | None:
    p = project_dir(pid) / "qa" / "pairs" / f"{qa_id}.json"
    if not p.exists():
        return None
    try:
        qa = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        log.warning("update_qa_pair(%s, %s): pair file corrupt/unreadable at %s", pid, qa_id, p, exc_info=True)
        return None
    for k, v in fields.items():
        qa[k] = v
    qa["updated_at"] = time.time()
    p.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    _forget_cached_pair(p.parent, p.name)
    return qa


def delete_qa_pair(pid: str, qa_id: str) -> bool:
    """Remove one pair file (a file reset drops its unreviewed pairs before a fresh mine)."""
    p = project_dir(pid) / "qa" / "pairs" / f"{qa_id}.json"
    if not p.exists():
        return False
    p.unlink()
    _forget_cached_pair(p.parent, p.name)
    return True


def delete_pairs_of_source(pid: str, source_id: str) -> int:
    """Delete every Q&A pair mined from one source (its file was purged). Returns how many were removed."""
    pairs = project_dir(pid) / "qa" / "pairs"
    removed = 0
    for qa in list_qa_pairs(pid, source_id=source_id):
        p = pairs / f"{qa['id']}.json"
        if p.is_file():
            p.unlink()
            _forget_cached_pair(pairs, p.name)
            removed += 1
    return removed
