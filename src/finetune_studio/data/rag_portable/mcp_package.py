"""Build a hostable, self-installing RAG package from a PortableRAG corpus.

The package is a plain directory (shipped as .tar.gz or .zip) that contains:

    <slug>-rag/
    ├── README.md                  human-readable quickstart
    ├── install.sh                 creates .venv + installs numpy (only dep)
    ├── run-http.sh                starts the REST search API
    ├── run-mcp.sh                 starts the MCP stdio server
    ├── requirements.txt           numpy
    ├── mcp-config.example.json    paste into Claude Desktop / OpenClaw / Cursor
    ├── server.py                  standalone search server (this repo's
    │                              standalone_server.py, copied verbatim)
    └── corpus/
        ├── manifest.json
        ├── chunks.jsonl           converted from chunks.parquet (stdlib-readable)
        ├── vectors.npy
        ├── vectors.idx.json
        ├── bm25.json
        └── sources/               original parsed text, for provenance

The recipient needs Python 3.10+ and nothing else: semantic search runs
against any OpenAI-compatible /v1/embeddings endpoint (LM Studio, Ollama,
OpenAI); without one the server still answers with keyword (BM25) search.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

_SERVER_SRC = Path(__file__).parent / "standalone_server.py"

_README_TMPL = """# {title} — searchable knowledge pack

Exported from Finetune Studio on {date}.
It is your project's documents, split into searchable chunks, with two
ready-made ways to query them: a plain HTTP API and an MCP server
(the format Claude Desktop, OpenClaw, Cursor and friends speak).

## What's inside

| Path | What it is |
|---|---|
| `corpus/` | the documents + search index (chunks, vectors, keyword index) |
| `server.py` | the search server — one file, reads `corpus/` |
| `install.sh` | creates a private Python venv and installs the only dependency (numpy) |
| `run-http.sh` | starts `http://localhost:8899/search?q=...` |
| `run-mcp.sh` | starts the MCP server on stdio (for agent apps) |
| `mcp-config.example.json` | config snippet for agent apps |

## Quickstart

```bash
bash install.sh          # once; makes .venv/
bash run-http.sh         # starts the search API on :8899
curl "http://localhost:8899/search?q=your+question&top_k=5"
```

Or one-shot from the shell, no server:

```bash
.venv/bin/python server.py --query "your question" --top-k 5
```

## Better results: point it at an embedding model

The index was built with **{embed_model}**. Without that model the server
still searches by keywords (BM25) — good, but meaning-search is better.
Any app that serves OpenAI-compatible `/v1/embeddings` works — LM Studio,
Ollama, OpenAI, vLLM:

```bash
export RAG_EMBED_BASE_URL="http://localhost:1234/v1"   # LM Studio default
# export RAG_EMBED_MODEL="..."   # optional; defaults to the model above
# export RAG_EMBED_API_KEY="..." # optional bearer token
bash run-http.sh
```

## Use it as an MCP server (agents)

- **Claude Desktop** — put this in `claude_desktop_config.json` (edit the path):

```
{{
  "mcpServers": {{
    "{slug}-rag": {{
      "command": "bash",
      "args": ["/ABSOLUTE/PATH/TO/{slug}-rag/run-mcp.sh"]
    }}
  }}
}}
```

- **OpenClaw** — `mcp.servers` in `openclaw.json`, same command/args shape.
- **Cursor / other MCP clients** — stdio command: `bash run-mcp.sh`.

Tools exposed: `rag_search(query, top_k)` and `rag_info()`.

## HTTP API

| Endpoint | Returns |
|---|---|
| `GET /health` | corpus name, chunk count, search mode |
| `GET /search?q=...&top_k=5` | ranked passages with source file names |
| `POST /search` (JSON `{{"query": "...", "top_k": 5}}`) | same |

## Notes

- Everything is local; the server binds your machine only, makes no calls
  except to the embedding endpoint you configure.
- `corpus/sources/` keeps the parsed text of every document, so you can
  always check where a passage came from.
- Re-export from Finetune Studio any time the documents change.
"""

_INSTALL_SH = """#!/usr/bin/env bash
# Creates a private venv and installs the only dependency (numpy).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
echo "Installed. Next:  bash run-http.sh   or   bash run-mcp.sh"
"""

_RUN_HTTP_SH = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || bash install.sh
exec .venv/bin/python server.py --http "${PORT:-8899}"
"""

_RUN_MCP_SH = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || bash install.sh
exec .venv/bin/python server.py --mcp
"""

_MCP_CONFIG_TMPL = """{{
  "mcpServers": {{
    "{slug}-rag": {{
      "command": "bash",
      "args": ["/ABSOLUTE/PATH/TO/{slug}-rag/run-mcp.sh"]
    }}
  }}
}}
"""


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "corpus").lower()).strip("-")
    return s[:48] or "corpus"


def _tar_data_filter(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    """Portable equivalent of tarfile's ``filter="data"`` (older Pythons
    don't accept the string form): normalize ownership, strip setuid bits,
    keep the executable bits our run scripts need."""
    if ti.issym() or ti.islnk():
        raise ValueError(f"refusing link member in package: {ti.name}")
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mode &= ~0o7000
    return ti


def build_package(
    corpus_dir: str | Path,
    out_path: str | Path,
    *,
    name: str | None = None,
    fmt: str = "tar.gz",
) -> Path:
    """Assemble the hostable package for a built corpus and archive it.

    ``corpus_dir`` must contain a built PortableRAG corpus (manifest.json +
    chunks.parquet + vectors.npy + vectors.idx.json + bm25.json).
    Returns the archive path. ``fmt``: ``tar.gz`` | ``zip``.
    """
    import pandas as pd  # local import: only needed at export time

    corpus = Path(corpus_dir)
    manifest_path = corpus / "manifest.json"
    for req in ("manifest.json", "chunks.parquet", "vectors.npy",
                "vectors.idx.json", "bm25.json"):
        if not (corpus / req).is_file():
            raise FileNotFoundError(f"corpus missing {req}: {corpus}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    title = name or manifest.get("name") or corpus.name
    slug = _slug(title)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    stage = Path(tempfile.mkdtemp(prefix="fts-rag-pkg-"))
    try:
        root = stage / f"{slug}-rag"
        (root / "corpus").mkdir(parents=True)
        shutil.copy2(manifest_path, root / "corpus" / "manifest.json")
        for f in ("vectors.npy", "vectors.idx.json", "bm25.json"):
            shutil.copy2(corpus / f, root / "corpus" / f)
        if (corpus / "sources").is_dir():
            shutil.copytree(corpus / "sources", root / "corpus" / "sources")
        # parquet -> jsonl so the recipient needs no pandas/pyarrow
        df = pd.read_parquet(corpus / "chunks.parquet")
        df.to_json(root / "corpus" / "chunks.jsonl",
                   orient="records", lines=True, force_ascii=False)
        shutil.copy2(_SERVER_SRC, root / "server.py")
        (root / "requirements.txt").write_text("numpy\n", encoding="utf-8")
        (root / "install.sh").write_text(_INSTALL_SH, encoding="utf-8")
        (root / "run-http.sh").write_text(_RUN_HTTP_SH, encoding="utf-8")
        (root / "run-mcp.sh").write_text(_RUN_MCP_SH, encoding="utf-8")
        (root / "mcp-config.example.json").write_text(
            _MCP_CONFIG_TMPL.format(slug=slug), encoding="utf-8")
        import time

        embed_model = str((manifest.get("embedding_model") or {}).get("name")
                          or "the embedding model named in corpus/manifest.json")
        (root / "README.md").write_text(_README_TMPL.format(
            title=title, slug=slug, embed_model=embed_model,
            date=time.strftime("%Y-%m-%d"),
        ), encoding="utf-8")
        for sh in ("install.sh", "run-http.sh", "run-mcp.sh"):
            (root / sh).chmod(0o755)

        if fmt == "zip":
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(root.rglob("*")):
                    if p.is_file():
                        zf.write(p, p.relative_to(stage))
        elif fmt in ("tar.gz", "tgz", "tar"):
            with tarfile.open(out, "w:gz" if fmt != "tar" else "w") as tf:
                tf.add(root, arcname=f"{slug}-rag", filter=_tar_data_filter)
        else:
            raise ValueError(f"unsupported package format: {fmt!r}")
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    log.info("rag package built: %s (%d bytes)", out, out.stat().st_size)
    return out
