"""BrainIndex — file-based corpus store.

Owns the directory layout (manifest, chunks, vectors, bm25, sources).
Build (parse + chunk + embed) and persist; load returns a BrainQuery.
Also: bundle/unbundle models for portability, export as tar/zip.
"""
from __future__ import annotations

import dataclasses
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
from finetune_studio.smart_brain.bm25 import BM25Index
from finetune_studio.smart_brain.constants import (
    DEFAULT_EMBEDDER,
    DEFAULT_RERANKER,
    EMBEDDER_LOCAL_PREFIX,
    RERANKER_LOCAL_PREFIX,
    SCHEMA_VERSION,
)
from finetune_studio.smart_brain.io import (
    read_json,
    relabel_imported_sources,
    try_import_pandas,
    write_json,
)

# Process-wide cached embedder (model_cache); keep the ``get_embedder`` name — tests patch it.
from finetune_studio.smart_brain.model_cache import (
    cached_embedder as get_embedder,
)
from finetune_studio.smart_brain.query import BrainQuery
from finetune_studio.smart_brain.schema import (
    ChunkSettings,
    IndexDocument,
    Manifest,
    RetrievalSettings,
)
from finetune_studio.smart_brain.shared_refs import resolve_model_ref
from finetune_studio.smart_brain.source_labels import display_name_for_path

_MODEL_DIR_NAMES = frozenset({"embedder", "reranker"})

log = logging.getLogger(__name__)


class BrainIndex:
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

    def read_manifest(self) -> Manifest:
        """The manifest alone. ``load()`` also reads every vector and loads the embedder (2.2 GB onto the GPU) — the
        settings route did that on every Smart Brain page visit just to show the chunk size (2026-10-19)."""
        return Manifest.from_json(read_json(self.manifest_path))

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
                              device: str = "auto",
                              progress=None) -> dict:
        """Parse the files of a plain directory, then index them (CLI / standalone corpora).

        Projects never come here: their files are parsed once by ``knowledge.sources`` and indexed through
        ``build_from_documents`` (``knowledge.index``). Always **replaces** the corpus content.
        """
        from finetune_studio.data.parsers import parse as parser_parse

        source_dir = Path(source_dir)
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)
        if extensions is None:
            from finetune_studio.data.parsers import readable_extensions
            extensions = sorted(readable_extensions())
        files: list[Path] = []
        for ext in extensions:
            files.extend(sorted(source_dir.rglob(f"*{ext}")))
            files.extend(sorted(source_dir.rglob(f"*{ext.upper()}")))
        files = list(dict.fromkeys(f for f in files if f.is_file()))
        docs: list[IndexDocument] = []
        for f in files:
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
            doc_id = hashlib.md5(f"{display_name}:{len(text)}:{f.stat().st_size}".encode()).hexdigest()[:12]
            docs.append(IndexDocument(doc_id, display_name, str(f), text))
        result = self.build_from_documents(docs, name=name, embedder=embedder, chunk_size=chunk_size,
                                           overlap=overlap, device=device, extra={"file_extensions": extensions})
        result["skipped"] = len(files) - result["documents"]
        return result

    def build_from_documents(self, docs: list[IndexDocument], *,
                             name: str | None = None,
                             embedder: str | None = None,
                             chunk_size: int = 400, overlap: int = 80,
                             device: str = "auto",
                             splitter: str | None = None,
                             extra: dict | None = None,
                             rag_settings: RetrievalSettings | None = None) -> dict:
        """Chunk, embed and BM25-index already parsed documents; write every corpus artifact.

        Always **replaces** the corpus content. No documents = nothing written (an existing corpus is kept), so a
        failed parse of every file never wipes a working index. ``splitter`` names pre-chunked input in the manifest
        (``passages``: a project's mining passages); ``extra`` lands in ``manifest.extra`` (e.g. what the
        index was built from, for stale detection).

        The retrieval settings carry over: ``rag_settings`` (a caller that removed the corpus first), else this
        corpus' own. No ``embedder`` = the one those settings chose, else ``DEFAULT_EMBEDDER`` (a rebuild used to
        reset every index to the default embedder and rerank/hybrid defaults).
        """
        from finetune_studio.smart_brain.chunking import chunk_documents

        docs = [d for d in docs if d.text.strip()]
        if not docs:
            return {"documents": 0, "chunks": 0, "skipped": 0}
        if rag_settings is None and self.manifest_path.exists():
            rag_settings = Manifest.from_json(read_json(self.manifest_path)).rag_settings
        embedder = embedder or (rag_settings.embedder if rag_settings else "") or DEFAULT_EMBEDDER
        encode, embed_info = get_embedder(name=resolve_model_ref(embedder, "embedder"), device=device)
        if embedder.startswith("shared:embedder:"):   # keep the portable ref, not this machine's path
            embed_info = dataclasses.replace(embed_info, name=embedder)
        # Chunks are sized in the embedder's own tokens when it exposes a tokenizer, else by a chars/token estimate.
        count_tokens = getattr(encode, "count_tokens", None)
        max_seq = int(getattr(encode, "max_seq_tokens", 0) or 0)
        if max_seq and chunk_size > max_seq:
            log.warning("chunk_size %d tokens exceeds the embedder's %d-token window: chunk tails will not be embedded",
                        chunk_size, max_seq)
        manifest = Manifest(
            name=name or self.dir.name,
            version=SCHEMA_VERSION,
            created_at=time.time(),
            updated_at=time.time(),
            embedding_model=embed_info,
            chunk_settings=ChunkSettings(size=chunk_size, overlap=overlap,
                                         splitter=splitter or ("rows+tokens" if count_tokens else "rows+chars")),
            rag_settings=(dataclasses.replace(rag_settings, embedder=embedder) if rag_settings else
                          RetrievalSettings(embedder=embedder, reranker=DEFAULT_RERANKER,
                                            rerank_enabled=True, hybrid_enabled=True)),
        )
        all_chunks = chunk_documents(docs, chunk_size, overlap, count_tokens)
        if not all_chunks:
            return {"documents": 0, "chunks": 0, "skipped": 0}
        if count_tokens and max_seq:
            too_long = sum(1 for c in all_chunks if count_tokens(c["text"]) > max_seq)
            if too_long:
                log.warning("%d chunk(s) exceed the embedder's %d-token window: their tails are not embedded",
                            too_long, max_seq)

        # Replace source artifacts to match the new corpus (no stale orphans).
        self._clear_source_artifacts()
        sources_dir = self.dir / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)
        all_documents: list[dict] = []
        for d in docs:
            (sources_dir / f"{d.document_id}.txt").write_text(d.text, encoding="utf-8")
            all_documents.append({"document_id": d.document_id, "source": d.source,
                                  "content_text_path": f"sources/{d.document_id}.txt", "filename": d.filename})

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
        manifest.extra = {**(extra or {}), "documents_meta": all_documents}
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
            "skipped": 0,
        }

    # ── Bundling: copy embedder + reranker into the corpus dir so the whole
    #     RAG is self-contained (no internet needed, no model download).
    #     Use case: ship the corpus to another machine. Unpack, load, query.
    # ──

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
            # Plaintext archive: never pick a world-readable default location.
            # User-facing exports use secure_bundle (encrypted) instead.
            raise ValueError("export_bundle needs an explicit out_path (plaintext archive)")
        out_path = Path(out_path)

        with tempfile.TemporaryDirectory(prefix="fts-brain-export-") as tmp:
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

        with tempfile.TemporaryDirectory(prefix="fts-brain-import-") as tmp:
            stage = Path(tmp) / "stage"
            stage.mkdir()

            name_lower = archive_path.name.lower()
            if name_lower.endswith((".tar.gz", ".tgz")):
                suffix = ".tar.gz"
            elif name_lower.endswith(".tar"):
                suffix = ".tar"
            elif name_lower.endswith(".zip"):
                suffix = ".zip"
            else:
                raise ValueError(
                    f"Unsupported archive format: {archive_path.suffix.lower()}"
                )
            if suffix in (".tar.gz", ".tgz"):
                with tarfile.open(archive_path, "r:gz") as tar:
                    # filter='data' rejects absolute paths / ../ traversal / device nodes
                    tar.extractall(stage, filter="data")
            elif suffix == ".tar":
                with tarfile.open(archive_path, "r") as tar:
                    tar.extractall(stage, filter="data")
            elif suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zf:
                    for member in zf.infolist():
                        target = (stage / member.filename).resolve()
                        if not target.is_relative_to(stage.resolve()):
                            raise ValueError(
                                f"Bundle contains unsafe path: {member.filename}"
                            )
                    zf.extractall(stage)

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
            relabel_imported_sources(self.dir)

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
- Chunking: {manifest.chunk_settings.splitter} (whole lines, table rows never split; embedder tokens, else estimated), size={manifest.chunk_settings.size}, overlap={manifest.chunk_settings.overlap}
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
from finetune_studio.smart_brain import BrainIndex
rag = BrainIndex(\"{d.name}\").load()
hits = rag.search("your question", top_k=5, hybrid=True, rerank=True)
```

## Migrate / re-embed

```bash
fts brain rebuild-vectors --corpus /path/to/corpus [--embedder NEW_MODEL]
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
        return BrainQuery(
            corpus_dir=self.dir, manifest=manifest, chunks=chunks_df,
            vectors=vectors, idx_map=idx_map, bm25=bm25, encode=encode,
        )

    def remove_source(self, source_id: str) -> bool:
        """Remove a single source — and its chunks/vectors/BM25 entries — by document id.

        Purges every row whose ``document_id`` matches *source_id* from
        chunks.parquet, vectors.npy, vectors.idx.json and bm25.json, and drops
        the matching entry from the manifest's ``documents_meta``. Without this,
        a "removed" source's chunks stayed fully searchable and still appeared
        in ``list_sources()`` — the DELETE endpoint reported success while the
        source remained live in search results.
        """
        removed = False
        sources_dir = self.dir / "sources"
        if sources_dir.exists():
            for f in sources_dir.glob(f"{source_id}*.txt"):
                f.unlink()
                removed = True

        if not self.chunks_path.exists():
            return removed

        pd = try_import_pandas()
        chunks_df = pd.read_parquet(self.chunks_path)
        if "document_id" not in chunks_df.columns:
            return removed
        keep_mask = (chunks_df["document_id"] != source_id).to_numpy()
        if keep_mask.all():
            return removed
        removed = True

        kept_df = chunks_df[keep_mask].reset_index(drop=True)
        kept_df.to_parquet(self.chunks_path, index=False)

        if self.vectors_path.exists():
            vectors = np.load(self.vectors_path)
            np.save(self.vectors_path, vectors[keep_mask])

        idx_map = {row["id"]: i for i, row in enumerate(kept_df.to_dict("records"))}
        write_json(self.idx_path, idx_map)

        bm25 = BM25Index.build(kept_df["text"].tolist())
        write_json(self.bm25_path, bm25.to_dict())

        if self.manifest_path.exists():
            manifest = Manifest.from_json(read_json(self.manifest_path))
            extra = manifest.extra or {}
            meta = extra.get("documents_meta")
            if isinstance(meta, list):
                extra["documents_meta"] = [
                    d for d in meta
                    if str(d.get("document_id") or d.get("id") or "") != source_id
                ]
                manifest.extra = extra
            manifest.documents = (
                int(kept_df["document_id"].nunique()) if len(kept_df) else 0
            )
            manifest.chunks = len(kept_df)
            manifest.updated_at = time.time()
            write_json(self.manifest_path, manifest.to_json())

        return removed

    def clear_sources(self) -> None:
        """Clear source text and every persisted index derived from it."""
        sources_dir = self.dir / "sources"
        if sources_dir.exists():
            for f in sources_dir.glob("*.txt"):
                f.unlink()

        pd = try_import_pandas()
        if self.chunks_path.exists():
            chunks_df = pd.read_parquet(self.chunks_path)
            chunks_df.iloc[0:0].to_parquet(self.chunks_path, index=False)
        if self.vectors_path.exists():
            vectors = np.load(self.vectors_path)
            np.save(self.vectors_path, vectors[:0])
        write_json(self.idx_path, {})
        write_json(self.bm25_path, BM25Index.build([]).to_dict())

        if self.manifest_path.exists():
            manifest = Manifest.from_json(read_json(self.manifest_path))
            extra = manifest.extra or {}
            extra["documents_meta"] = []
            extra["file_extensions"] = []
            manifest.extra = extra
            manifest.documents = 0
            manifest.chunks = 0
            manifest.updated_at = time.time()
            write_json(self.manifest_path, manifest.to_json())

    def list_sources(self) -> list[dict]:
        """List current corpus documents from manifest / chunks (not stale dirs).

        Never invents rows from orphaned ``sources/*.txt`` left by older builds.
        """
        from finetune_studio.smart_brain.source_labels import (
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

    def rebuild_vectors(self, embedder: str | None = None, device: str = "auto") -> dict:
        """Re-embed every chunk (and rebuild BM25); the documents and chunks stay.

        No ``embedder`` = the one the settings ask for (``rag_settings.embedder``: a settings change writes it there
        and says "rebuild vectors afterwards"), else the one the vectors were built with."""
        if not self.exists():
            raise FileNotFoundError(self.dir)
        pd = try_import_pandas()
        manifest = Manifest.from_json(read_json(self.manifest_path))
        new_name = embedder or manifest.rag_settings.embedder or manifest.embedding_model.name
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
