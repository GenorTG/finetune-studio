"""PortableRAG — file-based corpus store.

Owns the directory layout (manifest, chunks, vectors, bm25, sources).
Build (parse + chunk + embed) and persist; load returns a PortableRAGQuery.
Also: bundle/unbundle models for portability, export as tar/zip.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

from finetune_studio.data import shared_models as _sm
from finetune_studio.data.rag_portable.constants import (
    DEFAULT_EMBEDDER, DEFAULT_RERANKER, EMBEDDER_LOCAL_PREFIX, RERANKER_LOCAL_PREFIX,
    SCHEMA_VERSION,
)
from finetune_studio.data.rag_portable.embedders import get_embedder
from finetune_studio.data.rag_portable.shared_refs import resolve_model_ref
from finetune_studio.data.rag_portable.io import read_json, try_import_pandas, write_json
from finetune_studio.data.rag_portable.bm25 import BM25Index
from finetune_studio.data.rag_portable.query import PortableRAGQuery
from finetune_studio.data.rag_portable.schema import (
    ChunkSettings, EmbeddingModelInfo, Manifest, RagSettings,
)

log = logging.getLogger(__name__)


class PortableRAG:
    """File-based RAG corpus. One dir per corpus. All standard tools."""

    def __init__(self, corpus_dir: str | Path):
        self.dir = Path(corpus_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "sources").mkdir(exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"
        self.chunks_path = self.dir / "chunks.parquet"
        self.vectors_path = self.dir / "vectors.npy"
        self.idx_path = self.dir / "vectors.idx.json"
        self.bm25_path = self.dir / "bm25.json"

    def exists(self) -> bool:
        return (self.chunks_path.exists() and self.vectors_path.exists()
                and self.idx_path.exists() and self.manifest_path.exists())

    # ── Build / ingest ──

    def build_from_directory(self, source_dir: str | Path, *,
                              name: Optional[str] = None,
                              embedder: str = DEFAULT_EMBEDDER,
                              chunk_size: int = 400, overlap: int = 80,
                              extensions: Optional[list] = None,
                              device: str = "cpu",
                              progress=None) -> dict:
        """Parse files in source_dir, chunk, embed, build BM25, write all artifacts."""
        from finetune_studio.data.parsers import parse as parser_parse
        from finetune_studio.rag.ingest import chunk_text

        source_dir = Path(source_dir)
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)

        if extensions is None:
            from finetune_studio.data.parsers import PARSERS
            extensions = sorted(PARSERS.keys())

        encode, embed_info = get_embedder(name=embedder, device=device)
        manifest = Manifest(
            name=name or self.dir.name,
            version=SCHEMA_VERSION,
            created_at=time.time(),
            updated_at=time.time(),
            embedding_model=embed_info,
            chunk_settings=ChunkSettings(size=chunk_size, overlap=overlap),
            rag_settings=RagSettings(embedder=embedder, reranker=DEFAULT_RERANKER,
                                     rerank_enabled=True, hybrid_enabled=True),
        )

        # Recursive walk — accept both flat dirs and nested per-sha subdirs
        files = []
        for ext in extensions:
            files.extend(sorted(source_dir.rglob(f"*{ext}")))
            files.extend(sorted(source_dir.rglob(f"*{ext.upper()}")))
        # Dedup while preserving order
        seen = set()
        uniq_files = []
        for f in files:
            if f not in seen and f.is_file():
                seen.add(f)
                uniq_files.append(f)
        files = uniq_files
        if not files:
            return {"documents": 0, "chunks": 0, "skipped": 0}

        all_chunks: list[dict] = []
        all_documents: list[dict] = []

        for f in files:
            try:
                parsed = parser_parse(f)
            except Exception as e:
                if progress:
                    progress(f"skipping {f.name}: parse error {e}")
                continue
            text = parsed.get("text", "")
            if not text or not text.strip():
                continue
            doc_id = hashlib.md5(f"{f.name}:{len(text)}:{f.stat().st_size}".encode()).hexdigest()[:12]
            (self.dir / "sources" / f"{doc_id}.txt").write_text(text, encoding="utf-8")
            all_documents.append({"document_id": doc_id, "source": str(f),
                                 "content_text_path": f"sources/{doc_id}.txt"})
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap,
                                metadata={"source": str(f), "filename": f.name},
                                doc_id=doc_id)
            for c in chunks:
                all_chunks.append({
                    "id": c.id, "document_id": doc_id, "chunk_index": c.chunk_index,
                    "source": str(f), "filename": f.name, "text": c.text,
                    "metadata": c.metadata or {},
                })

        if not all_chunks:
            return {"documents": 0, "chunks": 0, "skipped": 0}

        texts = [c["text"] for c in all_chunks]
        vectors = encode(texts)
        pd = try_import_pandas()
        chunks_df = pd.DataFrame(all_chunks)
        chunks_df.to_parquet(self.chunks_path, index=False)
        np.save(self.vectors_path, vectors)
        idx_map = {c["id"]: i for i, c in enumerate(all_chunks)}
        write_json(self.idx_path, idx_map)

        # BM25 index
        bm25 = BM25Index.build([c["text"] for c in all_chunks])
        write_json(self.bm25_path, bm25.to_dict())

        manifest.documents = len(all_documents)
        manifest.chunks = len(all_chunks)
        manifest.extra = {
            "documents_meta": all_documents,
            "file_extensions": extensions,
        }
        write_json(self.manifest_path, manifest.to_json())

        # Register embedder + reranker in the SHARED model store, store refs
        # in manifest (NOT a local copy — share the model across corpora).
        try:
            emb_ref = _sm.register(embedder, kind="embedder", prefer_cached=True)
            rr_ref = _sm.register(DEFAULT_RERANKER, kind="reranker", prefer_cached=True)
            m = Manifest.from_json(read_json(self.manifest_path))
            m.embedding_model.name = emb_ref.to_compact()
            m.rag_settings.embedder = emb_ref.to_compact()
            m.rag_settings.reranker = rr_ref.to_compact()
            m.extra["shared_model_paths"] = {
                "embedder": emb_ref.path, "reranker": rr_ref.path,
            }
            write_json(self.manifest_path, m.to_json())
        except Exception as e:  # noqa: BLE001
            log.warning("shared model registration failed: %s", e)

        self._write_readme(manifest)

        return {
            "documents": manifest.documents,
            "chunks": manifest.chunks,
            "vector_dim": vectors.shape[1],
            "skipped": len(files) - manifest.documents,
        }

    # ── Bundling: copy embedder + reranker into the corpus dir so the whole
    #     RAG is self-contained (no internet needed, no model download).
    #     Use case: ship the corpus to another machine. Unpack, load, query.
    # ──

    def bundle_models(self, embedder_ref: Optional[str] = None,
                      reranker_ref: Optional[str] = None) -> dict:
        """Copy models from the SHARED store INTO the corpus dir for export.

        After this call, the corpus is self-contained: a `tar.gz` of the dir
        can be unpacked on a machine with the right Python packages and
        queries work fully offline.

        `embedder_ref` / `reranker_ref` use the magic prefix:
            "shared:embedder:<short_id>"  -> resolved via the shared store
        If omitted, the value currently in manifest.json is used.
        """
        out = {"embedder": None, "reranker": None, "size_bytes": 0}
        if not self.manifest_path.exists():
            raise FileNotFoundError("build the corpus first")
        m = Manifest.from_json(read_json(self.manifest_path))

        def _bundle_one(kind: str, current_ref: str, dest_name: str) -> Optional[str]:
            if not current_ref or not current_ref.startswith(f"shared:{kind}:"):
                return None
            short_id = current_ref[len(f"shared:{kind}:"):]
            src = _sm.resolve(short_id, kind)
            dest = self.dir / dest_name
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            for f in src.rglob("*"):
                if f.is_file():
                    shutil.copy2(f, dest / f.relative_to(src))
            abs_path = str(dest.resolve())
            if kind == "embedder":
                new_ref = f"{EMBEDDER_LOCAL_PREFIX}{abs_path}"
            else:
                new_ref = f"{RERANKER_LOCAL_PREFIX}{abs_path}"
            return new_ref

        new_emb = _bundle_one("embedder",
                              embedder_ref or m.embedding_model.name,
                              "embedder")
        new_rr = _bundle_one("reranker",
                             reranker_ref or m.rag_settings.reranker,
                             "reranker")
        if new_emb:
            m.embedding_model.name = new_emb
            m.rag_settings.embedder = new_emb
            out["embedder"] = new_emb
        if new_rr:
            m.rag_settings.reranker = new_rr
            out["reranker"] = new_rr
        m.updated_at = time.time()
        write_json(self.manifest_path, m.to_json())
        total = 0
        for p in self.dir.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        out["size_bytes"] = total
        return out

    def unshare_models(self) -> dict:
        """Reverse of bundle_models — switch manifest refs BACK to shared:
        references and remove the local copies. Use this when you don't need
        a self-contained copy anymore (e.g. you exported and now want to
        free disk space)."""
        out = {"embedder_removed": False, "reranker_removed": False, "bytes_freed": 0}
        if not self.manifest_path.exists():
            return out
        m = Manifest.from_json(read_json(self.manifest_path))
        # Re-register in shared store from local copy (it's faster than re-fetch)
        for kind, current, attr_embed, attr_rerank in [
            ("embedder", m.embedding_model.name, "name", None),
            ("reranker", m.rag_settings.reranker, None, "reranker"),
        ]:
            if current.startswith(f"{'embedder_local' if kind == 'embedder' else 'reranker_local'}:"):
                local_path = Path(current.split(":", 1)[1])
                if local_path.exists():
                    # Compute size before delete
                    sz = sum(p.stat().st_size for p in local_path.rglob("*") if p.is_file())
                    out["bytes_freed"] += sz
                    out[f"{kind}_removed"] = True
                    # Re-register into shared store
                    sm_ref = _sm.register(m.extra.get("shared_model_paths", {}).get(kind, "?"),
                                           kind=kind, src_dir=local_path, prefer_cached=True)
                    if kind == "embedder":
                        m.embedding_model.name = sm_ref.to_compact()
                        m.rag_settings.embedder = sm_ref.to_compact()
                    else:
                        m.rag_settings.reranker = sm_ref.to_compact()
                    shutil.rmtree(local_path)
        m.updated_at = time.time()
        write_json(self.manifest_path, m.to_json())
        return out

    def export_bundle(self, out_path: Optional[str | Path] = None,
                     fmt: str = "tar",
                     name: Optional[str] = None,
                     include_models: bool = True) -> Path:
        """Export the entire corpus as a self-contained archive.

        - `name`: filename stem (default: <corpus_dir_name>-bundle)
        - `fmt`: 'tar' (fast) or 'tar.gz' (compressed, slow for 2GB+)
        - `include_models`: if True (default for portability use case),
          copy the embedder + reranker from the shared store into the export
          bundle so the recipient can run it offline. If False, the bundle
          keeps `shared:...` references and is smaller but the recipient
          needs to either have the same shared store OR fetch the model.

        Returns the path to the written archive.
        """
        if out_path is None:
            safe = (name or f"{self.dir.name}-bundle").replace(" ", "_").replace("/", "_")
            ext = ".tar.gz" if fmt == "tar.gz" else ".tar"
            out_path = Path("/tmp") / f"{safe}{ext}"
        else:
            out_path = Path(out_path)

        # If including models, ensure they're locally bundled first
        if include_models:
            self.bundle_models()  # idempotent

        # Resolve symlinks so the archive contains real files (so unshared
        # / portable). Symlinks wouldn't survive being moved to another box.
        archive_root = self.dir
        symlinks_resolved = False
        # (our shared-store model has no symlinks — files are copied into corpus)

        if fmt == "tar.gz":
            if not str(out_path).endswith(".tar.gz"):
                out_path = out_path.with_suffix(".tar.gz")
            with tarfile.open(out_path, "w:gz") as tar:
                tar.add(str(archive_root), arcname=self.dir.name)
        elif fmt in ("tar", ""):
            if not str(out_path).endswith(".tar"):
                out_path = out_path.with_suffix(".tar")
            with tarfile.open(out_path, "w") as tar:
                tar.add(str(archive_root), arcname=self.dir.name)
        elif fmt == "zip":
            if not str(out_path).endswith(".zip"):
                out_path = out_path.with_suffix(".zip")
            with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in self.dir.rglob("*"):
                    if p.is_file():
                        zf.write(p, p.relative_to(self.dir.parent))
        else:
            raise ValueError(f"Unknown format: {fmt}")
        return out_path

    def _write_readme(self, manifest: Manifest) -> None:
        d = self.dir
        readme = f"""# {manifest.name} -- portable RAG corpus (v{manifest.version})

- Embedder: `{manifest.embedding_model.name}` ({manifest.embedding_model.dim}-dim, {'normalized' if manifest.embedding_model.normalize else 'raw'})
- Reranker: `{manifest.rag_settings.reranker}` (enabled={manifest.rag_settings.rerank_enabled})
- Hybrid retrieval: BM25 + dense (RRF k={manifest.rag_settings.rrf_k})
- Chunking: word-based, size={manifest.chunk_settings.size}, overlap={manifest.chunk_settings.overlap}
- Documents: {manifest.documents}
- Chunks: {manifest.chunks}

## Files

- `manifest.json` — corpus + retrieval settings
- `chunks.parquet` — chunk text + metadata
- `vectors.npy` — (N, {manifest.embedding_model.dim}) float32, L2-normalized dense embeddings
- `bm25.json` — sparse inverted index
- `vectors.idx.json` — {{chunk_id: row_index}}
- `sources/<doc_id>.txt` — original parsed text

## Search (4 lines)

```python
from finetune_studio.data.rag_portable import PortableRAG
rag = PortableRAG(\"{d.name}\").load()
hits = rag.search("your question", top_k=5, hybrid=True, rerank=True)
```

## Migrate / re-embed

```bash
python -m finetune_studio.data.rag rebuild-vectors /path/to/corpus [--embedder NEW_MODEL]
```
"""
        (d / "README.md").write_text(readme, encoding="utf-8")

    # ── Load ──

    def load(self):
        if not self.exists():
            raise FileNotFoundError(f"No RAG corpus at {self.dir} — build first.")
        pd = try_import_pandas()
        manifest = Manifest.from_json(read_json(self.manifest_path))
        chunks_df = pd.read_parquet(self.chunks_path)
        vectors = np.load(self.vectors_path).astype(np.float32)
        idx_map = read_json(self.idx_path)
        bm25 = BM25Index.from_dict(read_json(self.bm25_path))
        # Pass an absolute local path to the embedder/reranker if the manifest
        # stores one with the magic prefix. This survives tar+untar as long as
        # the absolute path on the destination is what we want — otherwise
        # the user can re-bundle or manually edit manifest.json.
        embed_name = resolve_model_ref(manifest.embedding_model.name, "embedder")
        corpus_dim = int(vectors.shape[1]) if getattr(vectors, "ndim", 0) == 2 else 0
        try:
            encode, embed_info = get_embedder(name=embed_name)
        except Exception as e:
            # Never silently fall back to a different embedder — a dim mismatch
            # (e.g. 384-d MiniLM corpus + 1024-d DEFAULT_EMBEDDER) crashes search.
            raise RuntimeError(
                f"Failed to load corpus embedder {embed_name!r} "
                f"(manifest dim={manifest.embedding_model.dim}, "
                f"vectors.npy dim={corpus_dim}). "
                f"Fix the shared/local embedder path or rebuild the corpus with "
                f"a matching embedder. Original error: {e}"
            ) from e
        loaded_dim = int(embed_info.dim)
        if corpus_dim and loaded_dim != corpus_dim:
            raise ValueError(
                f"Embedder dimension mismatch: {embed_name!r} produces "
                f"{loaded_dim}-d vectors but corpus vectors.npy is {corpus_dim}-d "
                f"(manifest recorded {manifest.embedding_model.dim}). "
                f"Rebuild with this embedder, or restore the original "
                f"{manifest.embedding_model.name!r} model."
            )
        return PortableRAGQuery(
            corpus_dir=self.dir, manifest=manifest, chunks=chunks_df,
            vectors=vectors, idx_map=idx_map, bm25=bm25, encode=encode,
        )

    def remove_source(self, source_id: str) -> bool:
        """Remove a single source from the corpus by its document id."""
        removed = False
        sources_dir = self.dir / "sources"
        if sources_dir.exists():
            for f in sources_dir.glob(f"{source_id}*.txt"):
                f.unlink()
                removed = True
        return removed

    def clear_sources(self) -> None:
        """Clear all sources from the corpus."""
        sources_dir = self.dir / "sources"
        if sources_dir.exists():
            for f in sources_dir.glob("*.txt"):
                f.unlink()
        # Reset manifest
        if self.manifest_path.exists():
            manifest = Manifest.from_json(read_json(self.manifest_path))
            manifest.updated_at = time.time()
            write_json(self.manifest_path, manifest.to_json())

    def list_sources(self) -> list[dict]:
        """List all sources in the corpus."""
        sources_dir = self.dir / "sources"
        if not sources_dir.exists():
            return []
        sources = []
        for f in sorted(sources_dir.glob("*.txt")):
            sources.append({
                "id": f.stem,
                "filename": f.stem,
                "size": f.stat().st_size,
            })
        return sources

    def rebuild_vectors(self, embedder: Optional[str] = None, device: str = "cpu") -> dict:
        if not self.exists():
            raise FileNotFoundError(self.dir)
        pd = try_import_pandas()
        manifest = Manifest.from_json(read_json(self.manifest_path))
        new_name = embedder or manifest.embedding_model.name
        new_name = resolve_model_ref(new_name, "embedder")
        encode, new_info = get_embedder(name=new_name, device=device)
        chunks_df = pd.read_parquet(self.chunks_path)
        vectors = encode(chunks_df["text"].tolist())
        np.save(self.vectors_path, vectors)
        # Rebuild BM25 too (cheap, keeps state consistent)
        bm25 = BM25Index.build(chunks_df["text"].tolist())
        write_json(self.bm25_path, bm25.to_dict())
        manifest.embedding_model = new_info
        manifest.rag_settings.embedder = new_name
        manifest.updated_at = time.time()
        write_json(self.manifest_path, manifest.to_json())
        return {"chunks": len(chunks_df), "vector_dim": vectors.shape[1],
                "embedder": new_name}
