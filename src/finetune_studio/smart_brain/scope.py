"""Which documents go into a package (its audience scope): all of them, or a chosen part.

A public chatbot package carries only ``public/``; an internal assistant everything. A document is in the package
when no selection is made (all), or when it matches any chosen folder/path prefix, any chosen tag, or is listed by
id; ``exclude`` prefixes then take documents out again. The package's corpus is rebuilt from the kept documents
only (``server_package.build_package(documents=...)``): passages, vectors, a BM25 index recomputed over them, and
their original files. Nothing outside the scope is in the archive.

A document's path is ``<folder>/<file name>`` for a project file in a folder (``<file name>`` otherwise); a
standalone ``--corpus`` index uses the file name.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


@dataclass(frozen=True)
class DocRef:
    document_id: str
    filename: str
    path: str
    tags: tuple[str, ...] = field(default=())


def _norm(prefix: str) -> str:
    return str(prefix or "").strip().lstrip("/").lower()


class DocumentScope(BaseModel):
    include: list[str] = Field(default_factory=list)       # path prefixes, e.g. "public/"
    exclude: list[str] = Field(default_factory=list)       # path prefixes taken out again
    tags: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)

    @field_validator("include", "exclude", "tags", "document_ids")
    @classmethod
    def _clean(cls, v: list[str]) -> list[str]:
        return [str(x).strip() for x in v if str(x).strip()]

    def is_all(self) -> bool:
        return not (self.include or self.exclude or self.tags or self.document_ids)

    def selects(self, doc: DocRef) -> bool:
        path = doc.path.lower()
        chosen = self.include or self.tags or self.document_ids
        hit = (not chosen
               or any(path.startswith(_norm(p)) for p in self.include)
               or bool({t.lower() for t in self.tags} & {t.lower() for t in doc.tags})
               or doc.document_id in self.document_ids)
        return hit and not any(path.startswith(_norm(p)) for p in self.exclude)

    def label(self) -> str:
        if self.is_all():
            return "all documents"
        parts = []
        if self.include:
            parts.append("paths " + ", ".join(self.include))
        if self.tags:
            parts.append("tags " + ", ".join(self.tags))
        if self.document_ids:
            parts.append(f"{len(self.document_ids)} chosen document(s)")
        text = " or ".join(parts) or "all documents"
        if self.exclude:
            text += " except " + ", ".join(self.exclude)
        return text


def _base(document_id: str) -> str:
    return str(document_id).split("~", 1)[0]


def _documents_meta(corpus_dir: Path) -> list[dict]:
    try:
        manifest = json.loads((Path(corpus_dir) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return list((manifest.get("extra") or {}).get("documents_meta") or [])


def documents(corpus_dir: str | Path, pid: str | None = None) -> list[DocRef]:
    """The index's documents (one per file; fact cards follow their file) with path and tags for choosing."""
    files: dict[str, dict] = {}
    if pid:
        from finetune_studio.data.fs import file_library as fl
        from finetune_studio.knowledge import sources

        paths = fl.folder_paths(pid)            # user folders only (not the by-type ones); a path: public/faq
        by_file = {f["id"]: dict(f, folder=paths.get(f["id"], "")) for f in fl.list_files(pid)}
        for s in sources.list_sources(pid):
            f = by_file.get(s.file_id)
            if f is not None:
                files[s.id] = f
    out: list[DocRef] = []
    seen: set[str] = set()
    for meta in _documents_meta(Path(corpus_dir)):
        did = _base(str(meta.get("document_id") or ""))
        if not did or did in seen:
            continue
        seen.add(did)
        f = files.get(did)
        name = str((f or {}).get("original_name") or meta.get("filename") or did)
        folder = str((f or {}).get("folder") or "")
        tags = tuple(t.strip() for t in str((f or {}).get("tags") or "").split(",") if t.strip())
        path = f"{folder}/{name}" if folder else str(meta.get("path") or "") if f is None else name
        out.append(DocRef(did, name, path or name, tags))
    return out


def resolve(scope: DocumentScope | None, docs: list[DocRef]) -> tuple[set[str] | None, dict]:
    """``(document ids to keep or None for all, the scope record for the manifest/README/rag_info)``."""
    scope = scope or DocumentScope()
    if scope.is_all():
        return None, {"label": "all documents", "documents": len(docs), "of": len(docs)}
    keep = {d.document_id for d in docs if scope.selects(d)}
    if not keep:
        raise ValueError(f"no document matches the package scope ({scope.label()})")
    return keep, {"label": scope.label(), "documents": len(keep), "of": len(docs), **scope.model_dump()}


def overview(docs: list[DocRef]) -> dict:
    """Folders and tags with their document counts, for the Export card's picker."""
    folders: dict[str, int] = {}
    tags: dict[str, int] = {}
    for d in docs:     # every folder level counts what is below it: public/ 40, public/faq/ 9
        parts = d.path.split("/")[:-1]
        for prefix in ([""] if not parts else ["/".join(parts[: i + 1]) + "/" for i in range(len(parts))]):
            folders[prefix] = folders.get(prefix, 0) + 1
        for t in d.tags:
            tags[t] = tags.get(t, 0) + 1
    return {"total": len(docs),
            "folders": [{"prefix": k, "label": k or "(no folder)", "count": v} for k, v in sorted(folders.items())],
            "tags": [{"tag": k, "count": v} for k, v in sorted(tags.items())],
            "documents": [{"document_id": d.document_id, "filename": d.filename, "path": d.path, "tags": list(d.tags)}
                          for d in docs]}


def probe_queries(corpus_dir: str | Path, excluded: set[str], limit: int = 3) -> dict[str, str]:
    """``document_id -> a few words of its first passage`` for documents left out: the self-test searches for them
    and expects no hit from outside the scope."""
    import pandas as pd

    out: dict[str, str] = {}
    df = pd.read_parquet(Path(corpus_dir) / "chunks.parquet", columns=["document_id", "text"])
    for did, text in zip(df["document_id"], df["text"]):
        base = _base(str(did))
        if base in excluded and base not in out:
            words = str(text).split()[:12]
            if words:
                out[base] = " ".join(words)
        if len(out) >= limit:
            break
    return out
