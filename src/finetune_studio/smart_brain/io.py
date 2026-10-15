"""JSON + pandas helpers.

Single responsibility: tiny I/O utilities shared by every other module in this package.
"""
from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, data: dict, indent: int = 2) -> None:
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def try_import_pandas():
    """Lazily import pandas, raising a clear error if it's missing."""
    try:
        import pandas as pd
        return pd
    except ImportError as e:
        raise RuntimeError(
            "pyarrow + pandas required for chunk parquet. Install: pip install pandas pyarrow"
        ) from e


def _relabel_source(src):
    """Absolute path from the exporting machine/project -> ``imported:<name>``."""
    if not isinstance(src, str) or src.startswith("imported:"):
        return src
    norm = src.replace("\\", "/")
    if norm.startswith("/") or (len(norm) > 2 and norm[1] == ":" and norm[2] == "/"):
        return "imported:" + (norm.rstrip("/").rsplit("/", 1)[-1] or norm)
    return src


def relabel_imported_sources(corpus_dir: Path) -> None:
    """Rewrite old-project absolute ``source`` paths in an imported corpus.

    Hits otherwise point at files of the exporting project, which do not exist
    here. Touches chunks.parquet and manifest ``extra.documents_meta`` only;
    vectors/BM25 are keyed by chunk id so retrieval is unaffected.
    """
    corpus_dir = Path(corpus_dir)
    chunks = corpus_dir / "chunks.parquet"
    if chunks.is_file():
        pd = try_import_pandas()
        df = pd.read_parquet(chunks)
        if "source" in df.columns:
            df["source"] = df["source"].map(_relabel_source)
            df.to_parquet(chunks, index=False)
    mpath = corpus_dir / "manifest.json"
    if mpath.is_file():
        raw = read_json(mpath)
        meta = (raw.get("extra") or {}).get("documents_meta")
        if isinstance(meta, list):
            for d in meta:
                if isinstance(d, dict) and "source" in d:
                    d["source"] = _relabel_source(d["source"])
            write_json(mpath, raw)
