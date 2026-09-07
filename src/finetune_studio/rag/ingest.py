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
import os
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
    return result.get("text", "")


def extract_pdf(path: Path) -> str:
    """Kept as a fallback for callers that want PDF-only extraction. Prefers
    pypdf, falls back to pdftotext, then OCR. Use extract_text() for the
    full multi-format pipeline.
    """
    try:
        from finetune_studio.data.parsers.pdf import parse as pdf_parse
        return pdf_parse(path).get("text", "")
    except Exception as e:  # noqa: BLE001
        return f"[extract_pdf failed: {e}]"


def extract_docx(path: Path) -> str:
    """Kept for backward-compat. Use extract_text() for full pipeline."""
    try:
        from finetune_studio.data.parsers.docx import parse as docx_parse
        return docx_parse(path).get("text", "")
    except Exception as e:  # noqa: BLE001
        return f"[extract_docx failed: {e}]"


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50,
               metadata: dict | None = None, doc_id: str = "") -> list[Chunk]:
    """Split text into overlapping word-based chunks. Each Chunk has a unique
    `id` and `document_id` so the vector store can dedupe and group."""
    if not text.strip():
        return []
    words = text.split()
    chunks: list[Chunk] = []
    start, idx = 0, 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_id = f"{doc_id}_{idx}" if doc_id else f"chunk_{idx}"
        chunks.append(Chunk(
            id=chunk_id,
            text=" ".join(words[start:end]),
            chunk_index=idx,
            document_id=doc_id,
            metadata=metadata or {},
        ))
        idx += 1
        start += chunk_size - overlap
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
            continue
        try:
            doc = ingest_file(str(path), chunk_size=chunk_size, overlap=overlap)
            if doc.chunks:
                out.append(doc)
        except Exception as e:  # noqa: BLE001
            print(f"[ingest] skipping {path.name}: {e}", flush=True)
    return out
