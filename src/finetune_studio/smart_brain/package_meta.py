"""Package versions: every package has a stable id, a version, a parent version and a document manifest.

The first export of a package gets a new ``package_id`` and version ``1.0.0``. "New version of an existing package"
keeps its id and bumps the minor version (``1.0.0`` -> ``1.1.0``; a typed version overrides it), and lists what
changed since the parent: documents added, changed (same path, different content) and removed. The record goes in
the package (``manifest["package"]`` inside the corpus, a name-free ``package.json`` beside it for ``update.sh``,
``CHANGES.md``) and in the project's package history (``brain-packages/history.json``), so a later export can pick
it as its base. A package opened from an archive (``package_import``) adds its own record to the history.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

HISTORY_NAME = "history.json"
FIRST_VERSION = "1.0.0"
_VERSION_RE = re.compile(r"^\d+(\.\d+){0,2}([-+][0-9A-Za-z.-]+)?$")


def new_package_id() -> str:
    return uuid.uuid4().hex[:16]


def valid_version(version: str) -> str:
    v = str(version or "").strip().lstrip("vV")
    if not _VERSION_RE.match(v):
        raise ValueError(f"version {version!r} is not like 1.2.0")
    return v


def bump_minor(version: str | None) -> str:
    """``1.4.2`` -> ``1.5.0``; ``2`` -> ``2.1.0``; nothing -> ``1.0.0``."""
    if not version:
        return FIRST_VERSION
    nums = [int(x) for x in re.findall(r"\d+", valid_version(version).split("-")[0].split("+")[0])[:3]]
    nums += [0] * (3 - len(nums))
    return f"{nums[0]}.{nums[1] + 1}.0"


def _version_key(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", str(v).split("-")[0])[:3])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def document_manifest(corpus_dir: str | Path, pid: str | None, keep: set[str] | None) -> list[dict]:
    """One row per document in the package: id, file name, path (folder/name), sha256 and size of its file.

    Project documents use the uploaded file (``knowledge.sources``); others the parsed text the index holds."""
    from .scope import documents as scope_documents

    corpus = Path(corpus_dir)
    sources = {}
    if pid:
        from finetune_studio.knowledge import sources as ks

        sources = {s.id: s for s in ks.list_sources(pid)}
    rows = []
    for d in scope_documents(corpus, pid):
        if keep is not None and d.document_id not in keep:
            continue
        s = sources.get(d.document_id)
        if s is not None and s.sha256:
            sha, size = s.sha256, int(s.size_bytes)
        else:
            text = corpus / "sources" / f"{d.document_id}.txt"
            sha, size = (_sha256(text), text.stat().st_size) if text.is_file() else ("", 0)
        rows.append({"document_id": d.document_id, "filename": d.filename, "path": d.path, "sha256": sha, "size": size})
    return rows


def changes(old: list[dict] | None, new: list[dict]) -> dict[str, list[dict]]:
    """Added / changed / removed documents, matched by path (a changed file gets a new id: ids come from content)."""
    before = {r["path"]: r for r in old or []}
    after = {r["path"]: r for r in new}
    return {
        "added": [after[p] for p in sorted(after) if p not in before],
        "changed": [after[p] for p in sorted(after) if p in before and before[p].get("sha256") != after[p].get("sha256")],
        "removed": [before[p] for p in sorted(before) if p not in after],
    }


def changes_markdown(record: dict, diff: dict[str, list[dict]], *, names: bool = True) -> str:
    """CHANGES.md. ``names=False`` (an encrypted package's plaintext copy) gives counts only."""
    head = f"# Changes — version {record['version']}\n\nPackage `{record['package_id']}`, built {record['created']}.\n\n"
    if not record.get("parent_version"):
        head += f"First version: {len(record['documents'])} document(s).\n"
    else:
        head += (f"Since version {record['parent_version']}: {len(diff['added'])} added, {len(diff['changed'])} "
                 f"changed, {len(diff['removed'])} removed.\n")
    if not names:
        return head + ("\nThe document names are encrypted with the corpus: `rag_info` (or `GET /health`) on the "
                       "running server lists them.\n")
    for key, title in (("added", "Added"), ("changed", "Changed"), ("removed", "Removed")):
        if diff[key] and (record.get("parent_version") or key != "added"):
            head += f"\n## {title}\n\n" + "".join(f"- {r['path']}\n" for r in diff[key])
    return head


def summary(diff: dict[str, list[dict]]) -> dict[str, list[str]]:
    return {k: [r["path"] for r in v] for k, v in diff.items()}


def make_record(*, base: dict | None, documents: list[dict], version: str | None = None) -> dict:
    """The new package's record: the base's id and a bumped version, or a new id at 1.0.0."""
    if version:
        version = valid_version(version)
        if base and _version_key(version) <= _version_key(base["version"]):
            raise ValueError(f"version {version} is not newer than {base['version']}")
    return {"package_id": base["package_id"] if base else new_package_id(),
            "version": version or bump_minor(base["version"] if base else None),
            "parent_version": base["version"] if base else None,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "documents": documents}


def package_filename(title: str, record: dict, *, ext: str, models: bool = False, encrypted: bool = True) -> str:
    """The archive's file name. It carries the package id, so two packages of one project (a public bot and the
    internal one) never share a name: with the bare ``<title>-smart-brain-v1.0.0`` the second export overwrote the
    first one's file while the history kept pointing both at it."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(title or "")).strip(".-")[:48] or "package"
    pid = re.sub(r"[^0-9a-f]", "", str(record.get("package_id") or ""))[:8]
    return (f"{safe}-smart-brain-{pid}-v{record['version']}{'-with-models' if models else ''}"
            f"{'' if encrypted else '-PLAINTEXT'}.{ext}")


def public_record(record: dict) -> dict:
    """``package.json``: no document names (it sits beside an encrypted corpus).

    ``policy`` (scope all/part, originals, hidden names, audience, encryption), ``widens`` (codes of what this version
    opened up over its parent, confirmed at export) and ``passphrase_changed`` let ``update.sh`` refuse an update
    that widens a running copy."""
    out = {"format": 1, **{k: record.get(k) for k in ("package_id", "version", "parent_version", "created")},
           "documents": len(record.get("documents") or [])}
    if record.get("policy"):
        out["policy"] = dict(record["policy"])
        out["widens"] = [w["code"] if isinstance(w, dict) else str(w) for w in record.get("widens") or []]
        out["passphrase_changed"] = bool(record.get("passphrase_changed"))
    return out


# ── the project's package history ──────────────────────────────────────────────────────────────────────────────

def package_dir(pid: str) -> Path:
    """Where a project's packages are written (relative to the studio's working directory, like ``output/``)."""
    return Path("output") / "projects" / pid / "brain-packages"


def history_path(package_dir: Path) -> Path:
    return Path(package_dir) / HISTORY_NAME


def read_history(package_dir: Path) -> list[dict]:
    try:
        data = json.loads(history_path(package_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [r for r in data if isinstance(r, dict) and r.get("package_id")] if isinstance(data, list) else []


def add_history(package_dir: Path, entry: dict) -> None:
    rows = [r for r in read_history(package_dir)
            if not (r["package_id"] == entry["package_id"] and r.get("version") == entry.get("version"))]
    rows.append(entry)
    path = history_path(package_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def latest(package_dir: Path, package_id: str) -> dict | None:
    rows = [r for r in read_history(package_dir) if r["package_id"] == package_id]
    return max(rows, key=lambda r: _version_key(r.get("version") or "0")) if rows else None


def _inherited(row: dict) -> dict[str, Any]:
    from .package_versions import base_settings

    return base_settings(row)


def packages(package_dir: Path) -> list[dict[str, Any]]:
    """One row per package id (its latest version first), newest package first, each with its versions."""
    by_id: dict[str, list[dict]] = {}
    for r in read_history(package_dir):
        by_id.setdefault(r["package_id"], []).append(r)
    out = []
    for pid_, rows in by_id.items():
        rows.sort(key=lambda r: _version_key(r.get("version") or "0"), reverse=True)
        out.append({"package_id": pid_, "name": rows[0].get("name") or "", "latest": rows[0]["version"],
                    "audience": rows[0].get("audience") or "", "scope": rows[0].get("scope") or "",
                    "documents": len(rows[0].get("documents") or []),
                    "versions": [{k: r.get(k) for k in ("version", "parent_version", "created", "filename", "origin",
                                                         "scope", "changes", "audience", "originals",
                                                         "hide_file_names", "encrypted", "widens")}
                                 | {"documents": len(r.get("documents") or [])}
                                 for r in rows],
                    "settings": _inherited(rows[0])})
    return sorted(out, key=lambda p: p["versions"][0].get("created") or "", reverse=True)
