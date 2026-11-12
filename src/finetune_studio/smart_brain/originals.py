"""The original files behind an index's documents, for a package that ships them ("Include original files").

A package normally holds the parsed text only. With originals on, every indexed document whose uploaded file is still
on disk ships that file too (``originals/<document_id>`` + an ``originals.json`` listing), inside the encrypted
container when the package is encrypted: anyone who can reach the package server can download them, so a public bot
leaves the option off.

Project indexes: document id = source id (``knowledge.sources``), the file is the source's raw upload. Standalone
``--corpus`` indexes: ``documents_meta[].source`` is the parsed file's own path. Fact-card rows (``<id>~facts``) share
their file's original.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import asdict, dataclass
from pathlib import Path

INDEX_NAME = "originals.json"
DIR_NAME = "originals"


@dataclass(frozen=True)
class OriginalFile:
    document_id: str
    filename: str
    path: Path
    mime_type: str
    size: int
    sha256: str
    date: float          # upload / modification time (unix seconds)

    def entry(self) -> dict:
        """The ``originals.json`` row (no local path: the package does not know where the studio kept it)."""
        row = asdict(self)
        row.pop("path")
        return row


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _mime(filename: str, given: str = "") -> str:
    return given or mimetypes.guess_type(filename)[0] or "application/octet-stream"


def _documents_meta(corpus_dir: Path) -> list[dict]:
    try:
        manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return list((manifest.get("extra") or {}).get("documents_meta") or [])


def base_id(document_id: str) -> str:
    """``<id>~facts`` (a file's fact cards) -> ``<id>``."""
    return str(document_id).split("~", 1)[0]


def collect(corpus_dir: str | Path, pid: str | None = None) -> list[OriginalFile]:
    """Every document of the index whose original file can be found, in index order (one entry per file)."""
    corpus = Path(corpus_dir)
    project_sources = {}
    if pid:
        from finetune_studio.knowledge import sources

        project_sources = {s.id: s for s in sources.list_sources(pid)}
    out: list[OriginalFile] = []
    seen: set[str] = set()
    for meta in _documents_meta(corpus):
        doc_id = base_id(str(meta.get("document_id") or ""))
        if not doc_id or doc_id in seen:
            continue
        src = project_sources.get(doc_id)
        if src is not None:
            path = Path(src.raw_path)
            if path.is_file():
                seen.add(doc_id)
                out.append(OriginalFile(doc_id, src.filename or path.name, path, _mime(src.filename, src.mime_type),
                                        path.stat().st_size, src.sha256 or _sha256(path), float(src.uploaded_at)))
            continue
        path = Path(str(meta.get("source") or ""))
        if pid or not path.is_file() or path.name == "parsed.txt":
            continue          # a project document without its upload, or a parsed-text path: nothing original
        seen.add(doc_id)
        name = str(meta.get("filename") or path.name)
        out.append(OriginalFile(doc_id, Path(name).name or path.name, path, _mime(name), path.stat().st_size,
                                _sha256(path), path.stat().st_mtime))
    return out
