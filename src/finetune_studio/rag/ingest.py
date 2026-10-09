"""Document ingestion — parse files and store chunks.

Now uses `finetune_studio.data.parsers` (33+ formats, OCR via tesseract).
Falls back to raw-text read for unknown extensions.

WHY THIS WAS REWRITTEN: the previous version only handled PDF and DOCX
and fell back to opening files as text — which means PNG/JPG were ingested
as raw binary, defeating OCR. All ingestion must go through the unified
parser package.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    id: str = ""
    path: str = ""
    filename: str = ""
    content: str = ""
    chunks: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    chunk_count: int = 0


@dataclass
class Chunk:
    id: str = ""
    text: str = ""
    chunk_index: int = 0
    document_id: str = ""
    metadata: dict = field(default_factory=dict)


def extract_text(file_path: str) -> str:
    """Extract text via the unified parser package.

    Routes through `finetune_studio.data.parsers` which has:
      - 33+ format-specific parsers
      - OCR fallback for image-only PDFs (tesseract)
      - Dedicated image parser for png/jpg/tiff/etc. (also tesseract)
      - Robust JSON, XML, CSV/TSV, EML parsers
    """
    from finetune_studio.data.parsers import parse as parser_parse
    result = parser_parse(Path(file_path))
    text = result.get("text", "")
    # Unknown extensions fall back to a lossy text read; binary content would
    # be indexed as garbage chunks, so refuse it with a clear message.
    if result.get("metadata", {}).get("parser") == "text_fallback" and "\x00" in text[:8192]:
        raise ValueError(
            f"{Path(file_path).name}: binary file with unsupported extension "
            f"'{Path(file_path).suffix}' — not ingested"
        )
    return text


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50,
               metadata: dict | None = None, doc_id: str = "",
               count_tokens: Callable[[str], int] | None = None) -> list[Chunk]:
    """Split text into overlapping, line-preserving chunks (``chunk_size``/``overlap`` in tokens).

    Table rows stay on their own line and are never split or merged with a neighbour (see
    ``data.rag_portable.chunking``). ``count_tokens`` is the embedder's tokenizer when the caller has one; without it a
    chars-per-token estimate is used. Each Chunk has a unique `id` and `document_id` so the vector store can dedupe and
    group."""
    from finetune_studio.data.rag_portable.chunking import chunk_rows

    chunks: list[Chunk] = []
    for idx, body in enumerate(chunk_rows(text, chunk_size, overlap, count_tokens)):
        chunks.append(Chunk(
            id=f"{doc_id}_{idx}" if doc_id else f"chunk_{idx}",
            text=body,
            chunk_index=idx,
            document_id=doc_id,
            metadata=metadata or {},
        ))
    return chunks


def ingest_file(file_path: str, chunk_size: int = 512, overlap: int = 50) -> Document:
    """Ingest a single file. Routes through extract_text() which uses the
    unified parser package (so OCR works for images and image-PDFs)."""
    path = Path(file_path)
    text = extract_text(str(path))
    file_id = hashlib.md5(f"{path.name}:{text[:100]}:{path.stat().st_size}".encode()).hexdigest()[:12]
    meta = {"source": str(path), "filename": path.name}
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap,
                       metadata=meta, doc_id=file_id)
    return Document(
        id=file_id,
        path=str(path),
        filename=path.name,
        content=text,
        chunks=chunks,
        metadata={**meta, "size_bytes": path.stat().st_size if path.exists() else 0},
        chunk_count=len(chunks),
    )


def ingest_directory(directory: str, chunk_size: int = 512, overlap: int = 50,
                     extensions: list | None = None) -> list[Document]:
    """Ingest all files in a directory whose extension is in the supported list."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    if extensions is None:
        # Default to all parser-supported formats
        from finetune_studio.data.parsers import PARSERS
        extensions = sorted(PARSERS.keys())
    out: list[Document] = []
    # Recursive walk: build corpus from flat OR nested directories.
    # Default behavior: rglob finds all files matching the extension list.
    files = []
    for ext in extensions:
        # rglob('*.ext') matches case-sensitively; do both lower/upper to be safe
        files.extend(sorted(dir_path.rglob(f"*{ext}")))
        files.extend(sorted(dir_path.rglob(f"*{ext.upper()}")))
    # Dedup while preserving order
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    for path in uniq:
        if not path.is_file():
            continue
        try:
            doc = ingest_file(str(path), chunk_size=chunk_size, overlap=overlap)
            if doc.chunks:
                out.append(doc)
        except Exception as e:  # noqa: BLE001
            print(f"[ingest] skipping {path.name}: {e}", flush=True)
    return out
