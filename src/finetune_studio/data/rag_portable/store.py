"""PortableRAG — file-based corpus store.

Owns the directory layout (manifest, chunks, vectors, bm25, sources).
Build (parse + chunk + embed) and persist; load returns a PortableRAGQuery.
Also: bundle/unbundle models for portability, export as tar/zip.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from finetune_studio.data import shared_models as _sm
from finetune_studio.data.rag_portable.bm25 import BM25Index
from finetune_studio.data.rag_portable.constants import (
    DEFAULT_EMBEDDER,
    DEFAULT_RERANKER,
    EMBEDDER_LOCAL_PREFIX,
    RERANKER_LOCAL_PREFIX,
    SCHEMA_VERSION,
)
from finetune_studio.data.rag_portable.embedders import get_embedder
from finetune_studio.data.rag_portable.io import (
    read_json,
    try_import_pandas,
    write_json,
)
from finetune_studio.data.rag_portable.query import PortableRAGQuery
from finetune_studio.data.rag_portable.schema import (
    ChunkSettings,
    Manifest,
    RagSettings,
)
from finetune_studio.data.rag_portable.shared_refs import resolve_model_ref
from finetune_studio.data.rag_portable.source_labels import (
    display_name_for_path,
    should_ingest_source_file,
)

_MODEL_DIR_NAMES = frozenset({"embedder", "reranker"})

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

    def _clear_source_artifacts(self) -> None:
        """Delete ``sources/*.txt`` so rebuild cannot leave stale orphans."""
        sources_dir = self.dir / "sources"
        if not sources_dir.exists():
            sources_dir.mkdir(parents=True, exist_ok=True)
            return
        for stale in sources_dir.glob("*.txt"):
            try:
                stale.unlink()
            except OSError:
                log.warning("failed to remove stale source artifact %s", stale)

    def build_from_directory(self, source_dir: str | Path, *,
                              name: str | None = None,
                              embedder: str = DEFAULT_EMBEDDER,
                              chunk_size: int = 400, overlap: int = 80,
                              extensions: list | None = None,
                              device: str = "cpu",
                              progress=None) -> dict:
        """Parse files in source_dir, chunk, embed, build BM25, write all artifacts.

        Always **replaces** corpus content (chunks, vectors, BM25, sources,
        manifest documents). This is not an append: every successful build
        rewrites the index from ``source_dir``. Callers that need a full
        directory wipe (bundled models, lockfiles, etc.) pass ``reset=True``
        on the HTTP build/rebuild routes before invoking this method.
        """
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
        # Collect parse results first so a no-op early return does not wipe
        # an existing corpus when every candidate fails ingest/parse.
        pending: list[tuple[Path, str, str, str]] = []

        for f in files:
            if not should_ingest_source_file(f):
                continue
            try:
                parsed = parser_parse(f)
            except Exception as e:  # noqa: BLE001
                if progress:
                    progress(f"skipping {f.name}: parse error {e}")
                continue
            text = parsed.get("text", "")
            if not text or not text.strip():
                continue
            display_name = display_name_for_path(f)
            doc_id = hashlib.md5(
                f"{display_name}:{len(text)}:{f.stat().st_size}".encode()
            ).hexdigest()[:12]
            pending.append((f, text, display_name, doc_id))

        if not pending:
            return {"documents": 0, "chunks": 0, "skipped": 0}

        # Replace source artifacts to match the new corpus (no stale orphans).
        self._clear_source_artifacts()
        sources_dir = self.dir / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)

        for f, text, display_name, doc_id in pending:
            (sources_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
            all_documents.append({"document_id": doc_id, "source": str(f),
                                 "content_text_path": f"sources/{doc_id}.txt",
                                 "filename": display_name})
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap,
                                metadata={"source": str(f), "filename": display_name},
                                doc_id=doc_id)
            for c in chunks:
                all_chunks.append({
                    "id": c.id, "document_id": doc_id, "chunk_index": c.chunk_index,
                    "source": str(f), "filename": display_name, "text": c.text,
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

    def bundle_models(self, embedder_ref: str | None = None,
                      reranker_ref: str | None = None) -> dict:
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

        def _bundle_one(kind: str, current_ref: str, dest_name: str) -> str | None:
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

    def export_bundle(self, out_path: str | Path | None = None,
                     fmt: str = "tar",
                     name: str | None = None,
                     include_models: bool = True) -> Path:
        """Export the corpus as an archive without mutating live corpus state.

        - `name`: filename stem (default: <corpus_dir_name>-bundle)
        - `fmt`: 'tar' (fast) or 'tar.gz' (compressed, slow for 2GB+)
        - `include_models`: if True, copy embedder + reranker into the *staged*
          export tree only. If False, the archive keeps `shared:...` refs and
          never includes ``embedder/`` or ``reranker/`` directories — even if a
          prior with-models export left those dirs on disk.

        Returns the path to the written archive.
        """
        if out_path is None:
            safe = (name or f"{self.dir.name}-bundle").replace(" ", "_").replace("/", "_")
            ext = ".tar.gz" if fmt == "tar.gz" else ".tar"
            out_path = Path("/tmp") / f"{safe}{ext}"
        else:
            out_path = Path(out_path)

        with tempfile.TemporaryDirectory(prefix="fts-rag-export-") as tmp:
            stage_root = Path(tmp) / self.dir.name
            self._stage_corpus_for_export(stage_root, include_models=include_models)
            out_path = self._archive_directory(stage_root, out_path, fmt=fmt)
        return out_path

    def import_bundle(
        self,
        archive_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> dict:
        """Import an exported corpus bundle into ``self.dir``.

        - ``archive_path``: path to a ``.tar``, ``.tar.gz`` or ``.zip`` produced
          by :meth:`export_bundle`.
        - ``overwrite``: if ``False`` (default), refuse to replace an existing
          corpus at ``self.dir``. If ``True``, remove it before extracting.

        Returns a dict with the loaded manifest stats. Raises ``FileNotFoundError``
        if the archive is missing, ``FileExistsError`` if the corpus already
        exists and ``overwrite`` is False, ``ValueError`` for unsupported
        formats or missing manifests.
        """
        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Bundle not found: {archive_path}")

        if self.exists() and not overwrite:
            raise FileExistsError(
                f"Corpus already exists at {self.dir}; pass overwrite=True to replace."
            )

        with tempfile.TemporaryDirectory(prefix="fts-rag-import-") as tmp:
            stage = Path(tmp) / "stage"
            stage.mkdir()

            suffix = archive_path.suffix.lower()
            if suffix in (".tar.gz", ".tgz"):
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(stage)
            elif suffix == ".tar":
                with tarfile.open(archive_path, "r") as tar:
                    tar.extractall(stage)
            elif suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(stage)
            else:
                raise ValueError(f"Unsupported archive format: {suffix}")

            # Find manifest.json. It may sit at stage/manifest.json or one
            # directory below (the export wraps everything in <corpus_name>/).
            candidates = [m for m in stage.rglob("manifest.json") if m.is_file()]
            if not candidates:
                raise ValueError(f"No manifest.json found in {archive_path}")
            # Prefer the manifest whose sibling is chunks.parquet — that's the
            # real corpus root, not an embedder/reranker sub-manifest.
            with_chunks = [m for m in candidates if (m.parent / "chunks.parquet").exists()]
            manifest_path = with_chunks[0] if with_chunks else candidates[0]
            corpus_root = manifest_path.parent

            manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
            for required in ("name", "embedding_model", "rag_settings", "documents", "chunks"):
                if required not in manifest_dict:
                    raise ValueError(f"Manifest missing required field: {required}")

            # Atomically replace the live corpus.
            if self.dir.exists():
                shutil.rmtree(self.dir)
            self.dir.mkdir(parents=True)

            for item in corpus_root.iterdir():
                dest = self.dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, symlinks=False)
                else:
                    shutil.copy2(item, dest)

            # Rewrite local model refs so they point at the freshly extracted
            # embedder/reranker directories inside self.dir, not the path the
            # source machine had.
            manifest = Manifest.from_json(json.loads(self.manifest_path.read_text()))
            for kind in ("embedder", "reranker"):
                prefix = EMBEDDER_LOCAL_PREFIX if kind == "embedder" else RERANKER_LOCAL_PREFIX
                if kind == "embedder":
                    current = manifest.embedding_model.name or manifest.rag_settings.embedder or ""
                else:
                    current = manifest.rag_settings.reranker or ""
                staged = self.dir / kind
                if current.startswith(prefix) and staged.exists():
                    new_ref = f"{prefix}{staged.resolve()}"
                    if kind == "embedder":
                        manifest.embedding_model.name = new_ref
                        manifest.rag_settings.embedder = new_ref
                    else:
                        manifest.rag_settings.reranker = new_ref
            manifest.updated_at = time.time()
            write_json(self.manifest_path, manifest.to_json())

        # Reload to verify the result is queryable, then return stats.
        loaded = self.load()
        manifest_dict = loaded.manifest.to_json()
        return {
            "corpus_dir": str(self.dir),
            "name": manifest_dict["name"],
            "documents": manifest_dict["documents"],
            "chunks": manifest_dict["chunks"],
            "embedder": manifest_dict["embedding_model"]["name"],
            "reranker": manifest_dict["rag_settings"]["reranker"],
            "version": manifest_dict.get("version"),
        }

    def _stage_corpus_for_export(
        self, stage: Path, *, include_models: bool
    ) -> None:
        """Copy corpus into *stage* for archiving; never mutates ``self.dir``."""
        stage.mkdir(parents=True, exist_ok=True)
        for item in self.dir.iterdir():
            if item.name in _MODEL_DIR_NAMES:
                # Always omit live staged models from the base copy; with-models
                # re-adds them below from shared/local sources into *stage* only.
                continue
            dest = stage / item.name
            if item.is_dir():
                shutil.copytree(item, dest, symlinks=False)
            elif item.is_file():
                shutil.copy2(item, dest)

        mpath = stage / "manifest.json"
        if not mpath.exists():
            return
        m = Manifest.from_json(read_json(mpath))
        shared_paths = dict((m.extra or {}).get("shared_model_paths") or {})

        if include_models:
            self._copy_models_into_stage(stage, m, shared_paths)
        else:
            self._ensure_shared_refs_in_manifest(m, shared_paths)
            write_json(mpath, m.to_json())

    def _copy_models_into_stage(
        self,
        stage: Path,
        m: Manifest,
        shared_paths: dict,
    ) -> None:
        """Copy embedder/reranker into *stage* and point manifest at them."""
        def _src_for(kind: str, current_ref: str) -> Path | None:
            if current_ref and current_ref.startswith(f"shared:{kind}:"):
                short_id = current_ref[len(f"shared:{kind}:"):]
                try:
                    return Path(_sm.resolve(short_id, kind))
                except Exception as e:  # noqa: BLE001
                    log.debug(
                        "shared %s resolve failed for %s: %s",
                        kind,
                        short_id,
                        e,
                    )
            if current_ref and current_ref.startswith(
                EMBEDDER_LOCAL_PREFIX if kind == "embedder" else RERANKER_LOCAL_PREFIX
            ):
                local = Path(current_ref.split(":", 1)[1])
                if local.exists():
                    return local
            live = self.dir / ("embedder" if kind == "embedder" else "reranker")
            if live.exists():
                return live
            hint = shared_paths.get(kind)
            if hint and Path(hint).exists():
                return Path(hint)
            return None

        emb_src = _src_for("embedder", m.embedding_model.name or m.rag_settings.embedder)
        rr_src = _src_for("reranker", m.rag_settings.reranker)

        if emb_src is not None:
            dest = stage / "embedder"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(emb_src, dest, symlinks=False)
            ref = f"{EMBEDDER_LOCAL_PREFIX}{dest.resolve()}"
            m.embedding_model.name = ref
            m.rag_settings.embedder = ref
        else:
            self._ensure_shared_refs_in_manifest(m, shared_paths, kinds=("embedder",))

        if rr_src is not None:
            dest = stage / "reranker"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(rr_src, dest, symlinks=False)
            ref = f"{RERANKER_LOCAL_PREFIX}{dest.resolve()}"
            m.rag_settings.reranker = ref
        else:
            self._ensure_shared_refs_in_manifest(m, shared_paths, kinds=("reranker",))

        m.updated_at = time.time()
        write_json(stage / "manifest.json", m.to_json())

    def _ensure_shared_refs_in_manifest(
        self,
        m: Manifest,
        shared_paths: dict,
        kinds: tuple[str, ...] = ("embedder", "reranker"),
    ) -> None:
        """Rewrite local model refs back to ``shared:...`` for slim exports."""
        for kind in kinds:
            if kind == "embedder":
                current = m.embedding_model.name or m.rag_settings.embedder or ""
                local_prefix = EMBEDDER_LOCAL_PREFIX
            else:
                current = m.rag_settings.reranker or ""
                local_prefix = RERANKER_LOCAL_PREFIX

            if current.startswith(f"shared:{kind}:"):
                continue

            shared_ref: str | None = None
            hint = shared_paths.get(kind)
            if hint:
                short_id = Path(hint).name
                if short_id:
                    shared_ref = f"shared:{kind}:{short_id}"
            if shared_ref is None and current.startswith(local_prefix):
                # Last resort: keep HF-style name if we never registered shared paths
                shared_ref = None

            if shared_ref:
                if kind == "embedder":
                    m.embedding_model.name = shared_ref
                    m.rag_settings.embedder = shared_ref
                else:
                    m.rag_settings.reranker = shared_ref
            elif current.startswith(local_prefix):
                # Drop absolute local path so the archive does not claim a
                # machine-specific embedder_local path without shipping weights.
                if kind == "embedder":
                    m.embedding_model.name = DEFAULT_EMBEDDER
                    m.rag_settings.embedder = DEFAULT_EMBEDDER
                else:
                    m.rag_settings.reranker = DEFAULT_RERANKER

        m.updated_at = time.time()

    def _archive_directory(
        self, archive_root: Path, out_path: Path, *, fmt: str
    ) -> Path:
        """Write *archive_root* to *out_path* in the requested format."""
        if fmt == "tar.gz":
            if not str(out_path).endswith(".tar.gz"):
                # with_suffix(".tar.gz") would yield ".tar.gz" incorrectly on
                # stems that already end in ".tar"; build the name explicitly.
                out_path = Path(str(out_path) + ".tar.gz")
            with tarfile.open(out_path, "w:gz") as tar:
                tar.add(str(archive_root), arcname=archive_root.name)
        elif fmt in ("tar", ""):
            if not str(out_path).endswith(".tar"):
                out_path = out_path.with_suffix(".tar")
            with tarfile.open(out_path, "w") as tar:
                tar.add(str(archive_root), arcname=archive_root.name)
        elif fmt == "zip":
            if not str(out_path).endswith(".zip"):
                out_path = out_path.with_suffix(".zip")
            parent = archive_root.parent
            with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in archive_root.rglob("*"):
                    if p.is_file():
                        zf.write(p, p.relative_to(parent))
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
        """List current corpus documents from manifest / chunks (not stale dirs).

        Never invents rows from orphaned ``sources/*.txt`` left by older builds.
        """
        from finetune_studio.data.rag_portable.source_labels import (
            prettify_source_label,
        )

        sources: list[dict] = []
        if self.manifest_path.exists():
            try:
                raw = read_json(self.manifest_path)
                meta = (raw.get("extra") or {}).get("documents_meta") or []
            except Exception:  # noqa: BLE001
                meta = []
            if meta:
                for d in meta:
                    did = str(d.get("document_id") or d.get("id") or "")
                    if not did:
                        continue
                    fname = prettify_source_label(
                        str(d.get("filename") or ""),
                        d.get("source"),
                    )
                    src_path = self.dir / "sources" / f"{did}.txt"
                    size = src_path.stat().st_size if src_path.is_file() else 0
                    sources.append({"id": did, "filename": fname, "size": size})
                return sources

        if not self.chunks_path.exists():
            return []
        try:
            pd = try_import_pandas()
            df = pd.read_parquet(self.chunks_path)
        except Exception:  # noqa: BLE001
            return []
        if len(df) == 0 or "document_id" not in df.columns:
            return []
        seen: set[str] = set()
        for _, row in df.iterrows():
            did = str(row.get("document_id") or "")
            if not did or did in seen:
                continue
            seen.add(did)
            fname = prettify_source_label(
                str(row.get("filename") or ""),
                row.get("source"),
            )
            src_path = self.dir / "sources" / f"{did}.txt"
            size = src_path.stat().st_size if src_path.is_file() else 0
            sources.append({"id": did, "filename": fname, "size": size})
        return sources

    def rebuild_vectors(self, embedder: str | None = None, device: str = "cpu") -> dict:
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
