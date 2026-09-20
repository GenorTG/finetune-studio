"""Build a hostable, self-installing RAG package from a PortableRAG corpus.

The package is a plain directory (shipped as .tar.gz or .zip) that contains:

    <slug>-rag/
    ├── README.md                  human-readable quickstart
    ├── install.sh                 creates .venv + installs deps (numpy; plus
    │                              torch-cpu + sentence-transformers when the
    │                              package bundles models)
    ├── run-http.sh                starts the REST search API
    ├── run-mcp.sh                 starts the MCP stdio server
    ├── requirements.txt           numpy (+ sentence-transformers if models)
    ├── mcp-config.example.json    paste into Claude Desktop / OpenClaw / Cursor
    ├── server.py                  standalone search server (this repo's
    │                              standalone_server.py, copied verbatim)
    └── corpus/
        ├── manifest.json
        ├── chunks.jsonl           converted from chunks.parquet (stdlib-readable)
        ├── vectors.npy
        ├── vectors.idx.json
        ├── bm25.json
        ├── embedder/              optional: bundled embedding model
        ├── reranker/              optional: bundled cross-encoder
        └── sources/               original parsed text, for provenance

With ``include_models=True`` the embedding model (and reranker) the corpus
was built with are copied into the package, so the recipient gets full
meaning-based semantic search with zero external services — a literal
self-deployable all-in-one RAG system. Without them the server still works:
keyword (BM25) search out of the box, semantic via any OpenAI-compatible
/v1/embeddings endpoint if one is configured.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import tarfile
import tempfile
import time
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
| `install.sh` | creates a private Python venv and installs the dependencies |
| `run-http.sh` | starts `http://localhost:8899/search?q=...` |
| `run-mcp.sh` | starts the MCP server on stdio (for agent apps) |
| `mcp-config.example.json` | config snippet for agent apps |
| `setup.sh` | one-command guided setup: install, port, start, optional service, prints your MCP entry |

## Quickstart

```bash
bash setup.sh            # guided: installs, picks a port, starts the server,
                         # can install a persistent service, and prints a
                         # copy-paste MCP entry with real paths
cat MCP-ENTRY.txt        # the MCP config setup.sh generated for this machine
```

Or do it by hand:

```bash
bash install.sh          # once; makes .venv/ (numpy; + torch/ST for model packs)
bash run-http.sh         # starts the search API on :8899 (PORT=… to change)
curl "http://localhost:8899/search?q=your+question&top_k=5"
```

Or one-shot from the shell, no server:

```bash
.venv/bin/python server.py --query "your question" --top-k 5
```

{embed_section}

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
  except to the embedding endpoint you configure (and none at all when the
  model is bundled).
- `corpus/sources/` keeps the parsed text of every document, so you can
  always check where a passage came from.
- Re-export from Finetune Studio any time the documents change.
"""

_README_SECTION_API = """## Better results: point it at an embedding model

The index was built with **{embed_model}**. Without that model the server
still searches by keywords (BM25) — good, but meaning-search is better.
Any app that serves OpenAI-compatible `/v1/embeddings` works — LM Studio,
Ollama, OpenAI, vLLM:

```bash
export RAG_EMBED_BASE_URL="http://localhost:1234/v1"   # LM Studio default
# export RAG_EMBED_MODEL="..."   # optional; defaults to the model above
# export RAG_EMBED_API_KEY="***" # optional bearer token
bash run-http.sh
```

"""

_README_SECTION_OFFLINE = """## Full semantic search is already inside

This package bundles the embedding model the index was built with
(**{embed_model}**) in `corpus/embedder/` — meaning-based search works
completely offline: no API keys, no provider, no internet. If a reranker
was bundled too (`corpus/reranker/`), results get a second scoring pass
automatically.

`install.sh` sets up the venv for this (numpy + sentence-transformers; the
torch CPU wheel is the big download, a few minutes once). The first search
loads the model into RAM (~1-2 GB) and takes a second or two; after that it
is fast. CPU is the default — set `RAG_DEVICE=cuda` to use a GPU.

"""

_INSTALL_SH = """#!/usr/bin/env bash
# Creates a private venv and installs the dependencies (see requirements.txt).
# Model packages (corpus/embedder/ present) get torch-CPU + sentence-transformers.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
if [ -d corpus/embedder ]; then
  echo "Model package: installing torch (CPU) + sentence-transformers — a few minutes, once…"
  .venv/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi
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

_SETUP_SH = """#!/usr/bin/env bash
# One-command setup for this Finetune Studio RAG pack.
#
#   bash setup.sh              guided: port, start server, optional service
#   bash setup.sh --yes        defaults only (install + MCP entry, no prompts)
#   bash setup.sh --yes --start --port 8899
#   bash setup.sh --service    run as a persistent systemd user service
#   bash setup.sh --uninstall  remove the service (keeps files)
#
# Always writes MCP-ENTRY.txt with a copy-pasteable MCP config that has
# this machine's absolute paths already filled in.
set -euo pipefail
cd "$(dirname "$0")"
PKG_DIR="$(pwd)"
NAME="$(basename "$PKG_DIR")"
UNIT="fts-rag-$NAME.service"
UNIT_PATH="$HOME/.config/systemd/user/$UNIT"
PORT=8899
ASSUME_YES=0; DO_START=0; DO_SERVICE=0; DO_UNINSTALL=0; PORT_SET=0

while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y)      ASSUME_YES=1 ;;
    --port)        shift; PORT="${1:?--port needs a number}"; PORT_SET=1 ;;
    --port=*)      PORT="${1#--port=}"; PORT_SET=1 ;;
    --start)       DO_START=1 ;;
    --service)     DO_SERVICE=1 ;;
    --uninstall)   DO_UNINSTALL=1 ;;
    -h|--help)     sed -n '2,12p' "$0" | sed 's/^# \\?//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)"; exit 2 ;;
  esac
  shift
done

ask_yn() {  # ask_yn "prompt" default(Y|N) -> 0=yes
  local prompt="$1" def="${2:-N}" ans
  if [ "$ASSUME_YES" = 1 ] || [ ! -t 0 ]; then [ "$def" = "Y" ]; return; fi
  read -r -p "$prompt [$def] " ans || ans=""
  ans="${ans:-$def}"
  case "$ans" in [Yy]*) return 0 ;; *) return 1 ;; esac
}

if [ "$DO_UNINSTALL" = 1 ]; then
  if command -v systemctl >/dev/null 2>&1 && [ -f "$UNIT_PATH" ]; then
    systemctl --user disable --now "$UNIT" 2>/dev/null || true
    rm -f "$UNIT_PATH"
    systemctl --user daemon-reload
    echo "Service '$UNIT' removed."
  else
    echo "No service installed at $UNIT_PATH — nothing to remove."
  fi
  if [ -f rag-server.pid ]; then
    kill "$(cat rag-server.pid)" 2>/dev/null || true
    rm -f rag-server.pid
    echo "Stopped the locally started server."
  fi
  exit 0
fi

# 1) dependencies (venv)
if [ ! -x .venv/bin/python ]; then
  echo "Creating venv + installing dependencies…"
  bash install.sh
fi

# 2) port (prompt only when interactive)
if [ "$PORT_SET" = 0 ] && [ "$ASSUME_YES" = 0 ] && [ -t 0 ]; then
  read -r -p "Port for the search API [$PORT]: " ans || ans=""
  PORT="${ans:-$PORT}"
fi
case "$PORT" in (*[!0-9]*|'') PORT=8899 ;; esac

# 3) copy-paste MCP entry (absolute paths, ready to paste)
cat > MCP-ENTRY.txt <<ENTRY
{
  "mcpServers": {
    "$NAME": {
      "command": "bash",
      "args": ["$PKG_DIR/run-mcp.sh"]
    }
  }
}
ENTRY
echo
 echo "=== Copy-paste MCP entry (also saved to MCP-ENTRY.txt) ==="
cat MCP-ENTRY.txt
echo "=== Paste into Claude Desktop / OpenClaw mcp.servers / Cursor; or use HTTP: ==="
echo "  curl 'http://localhost:$PORT/search?q=your+question&top_k=5'"
echo

HAVE_SYSTEMD=0
if command -v systemctl >/dev/null 2>&1 && [ -n "${XDG_RUNTIME_DIR:-}" ]; then HAVE_SYSTEMD=1; fi

# 4) persistent service?
if [ "$DO_SERVICE" = 1 ] || ask_yn "Install as a service that stays up (starts on login, restarts on crash)?" N; then
  if [ "$HAVE_SYSTEMD" = 0 ]; then
    echo "No systemd user session here — skipping service. Start manually with: bash run-http.sh"
  else
    ENV_LINES=""
    [ -n "${RAG_EMBED_BASE_URL:-}" ] && ENV_LINES="Environment=RAG_EMBED_BASE_URL=$RAG_EMBED_BASE_URL"
    [ -n "${RAG_EMBED_API_KEY:-}" ] && ENV_LINES="$ENV_LINES
Environment=RAG_EMBED_API_KEY=$RAG_EMBED_API_KEY"
    mkdir -p "$(dirname "$UNIT_PATH")"
    cat > "$UNIT_PATH" <<UNIT
[Unit]
Description=Finetune Studio RAG pack ($NAME)
After=network-online.target

[Service]
WorkingDirectory=$PKG_DIR
ExecStart=$PKG_DIR/.venv/bin/python $PKG_DIR/server.py --http $PORT
$ENV_LINES
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
UNIT
    systemctl --user daemon-reload
    systemctl --user enable --now "$UNIT"
    echo "Service installed: $UNIT (port $PORT)"
    echo "  status:  systemctl --user status $UNIT"
    echo "  logs:    journalctl --user -u $UNIT -f"
    echo "  remove:  bash setup.sh --uninstall"
    if ! loginctl show-user "$(id -un)" 2>/dev/null | grep -q 'Linger=yes'; then
      echo "  tip: to keep it running after logout run: loginctl enable-linger $(id -un)"
    fi
    exit 0
  fi
fi

# 5) or just run it now
if [ "$DO_START" = 1 ] || ask_yn "Start the search server now on port $PORT?" Y; then
  nohup .venv/bin/python server.py --http "$PORT" > rag-server.log 2>&1 &
  echo $! > rag-server.pid
  sleep 2
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "Search server running on http://localhost:$PORT (log: rag-server.log, stop: kill \\$(cat rag-server.pid))"
  else
    echo "Server started but /health not answering yet — check rag-server.log"
  fi
else
  echo "Not started. Whenever you like: bash run-http.sh  (PORT=… to change the port)"
fi
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


_SKIP_PARTS = frozenset({"__pycache__", ".locks"})


def _copy_shared_model(ref: str, kind: str, dest: Path) -> str | None:
    """Copy a ``shared:<kind>:<short_id>`` model dir into ``dest``.

    Returns the short id on success, None when the ref is not a shared-store
    reference. Lock files and caches are skipped — the recipient re-creates
    those locally if it ever needs to.
    """
    prefix = f"shared:{kind}:"
    ref = str(ref or "")
    if not ref.startswith(prefix):
        return None
    from finetune_studio.data import shared_models as sm
    short_id = ref[len(prefix):]
    src = sm.resolve(short_id, kind)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        if _SKIP_PARTS & set(rel.parts) or f.name.endswith(".lock"):
            continue
        tgt = dest / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, tgt)
    return short_id


def build_package(
    corpus_dir: str | Path,
    out_path: str | Path,
    *,
    name: str | None = None,
    fmt: str = "tar.gz",
    include_models: bool = False,
) -> Path:
    """Assemble the hostable package for a built corpus and archive it.

    ``corpus_dir`` must contain a built PortableRAG corpus (manifest.json +
    chunks.parquet + vectors.npy + vectors.idx.json + bm25.json).
    Returns the archive path. ``fmt``: ``tar.gz`` | ``zip``.
    ``include_models`` copies the corpus's shared embedder (+ reranker)
    into the package for fully offline semantic search.
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
        for f in ("vectors.npy", "vectors.idx.json", "bm25.json"):
            shutil.copy2(corpus / f, root / "corpus" / f)
        if (corpus / "sources").is_dir():
            shutil.copytree(corpus / "sources", root / "corpus" / "sources")
        # parquet -> jsonl so the recipient needs no pandas/pyarrow
        df = pd.read_parquet(corpus / "chunks.parquet")
        df.to_json(root / "corpus" / "chunks.jsonl",
                   orient="records", lines=True, force_ascii=False)

        embed_model = str((manifest.get("embedding_model") or {}).get("name")
                          or "the embedding model named in corpus/manifest.json")
        requirements = "numpy\n"
        embed_section = _README_SECTION_API
        if include_models:
            rs = manifest.setdefault("rag_settings", {})
            emb_id = _copy_shared_model(
                (manifest.get("embedding_model") or {}).get("name", ""),
                "embedder", root / "corpus" / "embedder")
            if not emb_id:
                raise FileNotFoundError(
                    "include_models requested but this corpus does not "
                    "reference a shared embedder — rebuild it once from the "
                    "RAG page, then export with models")
            rr_id = _copy_shared_model(rs.get("reranker", ""), "reranker",
                                       root / "corpus" / "reranker")
            # Rewrite refs for display only — server.py loads bundled models
            # by directory, never by these paths.
            manifest["embedding_model"]["name"] = emb_id
            rs["embedder"] = emb_id
            if rr_id:
                rs["reranker"] = rr_id
            embed_model = emb_id
            requirements = "numpy\nsentence-transformers>=3.0\n"
            embed_section = _README_SECTION_OFFLINE
        (root / "corpus" / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        shutil.copy2(_SERVER_SRC, root / "server.py")
        (root / "requirements.txt").write_text(requirements, encoding="utf-8")
        (root / "install.sh").write_text(_INSTALL_SH, encoding="utf-8")
        (root / "run-http.sh").write_text(_RUN_HTTP_SH, encoding="utf-8")
        (root / "run-mcp.sh").write_text(_RUN_MCP_SH, encoding="utf-8")
        (root / "mcp-config.example.json").write_text(
            _MCP_CONFIG_TMPL.format(slug=slug), encoding="utf-8")
        (root / "setup.sh").write_text(_SETUP_SH, encoding="utf-8")
        (root / "README.md").write_text(_README_TMPL.format(
            title=title, slug=slug, embed_model=embed_model,
            date=time.strftime("%Y-%m-%d"),
            embed_section=embed_section.format(embed_model=embed_model),
        ), encoding="utf-8")
        for sh in ("install.sh", "run-http.sh", "run-mcp.sh", "setup.sh"):
            (root / sh).chmod(0o755)

        if fmt == "zip":
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(root.rglob("*")):
                    if p.is_file():
                        zf.write(p, p.relative_to(stage))
        elif fmt in ("tar.gz", "tgz", "tar"):
            gz = fmt != "tar"
            # safetensors weights barely compress; level 9 on a 2 GB model
            # package wastes minutes. Text-only packs keep level 9.
            kwargs = {"compresslevel": 1 if include_models else 9} if gz else {}
            with tarfile.open(out, "w:gz" if gz else "w", **kwargs) as tf:
                tf.add(root, arcname=f"{slug}-rag", filter=_tar_data_filter)
        else:
            raise ValueError(f"unsupported package format: {fmt!r}")
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    log.info("rag package built: %s (%d bytes)", out, out.stat().st_size)
    return out
