"""Writer for the chunks/ directory + manifest.

Single responsibility: take a list of chunk strings + optional per-chunk meta,
write them to files/<sha>/chunks/NNNN.txt, and write a manifest.json.
"""
from __future__ import annotations

import json
from typing import Optional

from finetune_studio.data.fs.paths import file_dir


def write_chunks(pid: str, sha256: str, chunks: list[str], chunk_meta: Optional[list[dict]] = None) -> None:
    fd = file_dir(pid, sha256)
    chunks_dir = fd / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    # Clear any old chunks (only safe because we only ever write a fresh parse)
    for old in chunks_dir.glob("*.txt"):
        old.unlink()
    manifest = []
    for i, text in enumerate(chunks):
        chunk_path = chunks_dir / f"{i:04d}.txt"
        chunk_path.write_text(text, encoding="utf-8")
        manifest.append({
            "index": i,
            "char_count": len(text),
            "source_section": (chunk_meta or [{}] * len(chunks))[i].get("source_section", ""),
            "chunk_path": str(chunk_path.relative_to(fd.parent)),
        })
    (chunks_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
