"""Build a hostable, self-installing RAG package from a PortableRAG corpus.

The package (``.tar.gz`` / ``.tar`` / ``.zip``) contains::

    <root>/
    ├── README.md                  quickstart + config + security notes
    ├── install.sh                 creates .venv + installs deps; takes the
    │                              runtime flags and saves them to rag.config.json
    ├── run-http.sh / run-mcp.sh   start the REST / MCP stdio server (flags pass through)
    ├── setup.sh                   guided setup (+ optional systemd user service)
    ├── requirements.txt           numpy (+ cryptography, + sentence-transformers)
    ├── rag.config.json            runtime defaults written at export time
    │                              (flags > env RAG_* > this file > built-ins)
    ├── mcp-config.example.json
    ├── server.py                  standalone_server.py, verbatim
    ├── rag_container.py           encrypted-container reader (encrypted packs)
    └── corpus/
        ├── corpus.enc             ENCRYPTED: manifest, chunks, vectors, BM25,
        │                          document names/metadata, original sources
        │                          (AES-256-GCM, scrypt-from-passphrase key)
        ├── embedder/              optional bundled embedding model (public weights)
        └── reranker/              optional bundled cross-encoder   (public weights)

Encryption is ON by default.  The key is never shipped or stored: it is
derived from a passphrase the exporter supplies (or that is generated and
returned once).  ``encrypt=False`` produces the legacy plaintext layout
(manifest.json / chunks.jsonl / vectors.npy / ... under ``corpus/``).

The build never stages plaintext on disk: content is read into memory,
encrypted frame by frame into an anonymous (unlinked) temp file that only
ever holds ciphertext, and streamed into the archive.  Model files are
streamed straight from the shared store into the archive.
"""
from __future__ import annotations

import contextlib
import json
import logging
import re
import secrets
import shutil
import tarfile
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from .export_config import RUNTIME_CONFIG_NAME, RagExportConfig

log = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_SERVER_SRC = _HERE / "standalone_server.py"
_CONTAINER_SRC = _HERE / "rag_container.py"
ENC_NAME = "corpus.enc"
MIN_PASSPHRASE_LEN = 8
NEUTRAL_ROOT = "rag-pack"          # encrypted packs never put names in paths
_COPY_BUF = 1024 * 1024


@dataclass
class PackageResult:
    """What ``build_package`` produced.  ``passphrase`` is set ONLY when it was
    generated here (so the caller can show it once); it is excluded from repr
    and must never be persisted or logged."""

    path: Path
    encrypted: bool
    root: str
    size: int
    passphrase: str | None = field(default=None, repr=False)


_README_TMPL = """# {title} — searchable knowledge pack

Exported from Finetune Studio on {date}.
It is a set of documents, split into searchable chunks, with two ready-made
ways to query them: a plain HTTP API and an MCP server (the format Claude
Desktop, OpenClaw, Cursor and friends speak).

## What's inside

| Path | What it is |
|---|---|
| `corpus/` | the documents + search index{enc_note} |
| `server.py` | the search server — one file{container_note} |
| `rag.config.json` | runtime defaults baked in at export (see "Configuration") |
| `install.sh` | creates a private Python venv and installs the dependencies |
| `run-http.sh` | starts the search API (default `http://127.0.0.1:8899`) |
| `run-mcp.sh` | starts the MCP server on stdio (for agent apps) |
| `mcp-config.example.json` | config snippet for agent apps |
| `setup.sh` | guided setup: install, port, start, optional service, MCP entry |

{security_section}
## Quickstart

```bash
bash setup.sh            # guided: installs, picks a port, starts the server,
                         # can install a persistent service, and prints a
                         # copy-paste MCP entry with real paths
cat MCP-ENTRY.txt
```

Or by hand:

```bash
bash install.sh          # once; makes .venv/
bash run-http.sh         # search API on 127.0.0.1:8899
{curl_hint}
```

One-shot from the shell, no server:

```bash
.venv/bin/python server.py --query "your question" --top-k 5
```

## Configuration

Precedence (highest first): **command-line flags > environment (`RAG_*`) >
`rag.config.json` > built-in defaults.**

| Setting | Flag | Env | Default here |
|---|---|---|---|
| bind address | `--host` | `RAG_HOST` | `{host}` |
| port | `--port` | `RAG_PORT` | `{port}` |
| results per search | `--top-k` | `RAG_TOP_K` | `{top_k}` |
| device for bundled models | `--device` | `RAG_DEVICE` | `{device}` |
| reranker | `--reranker` / `--no-reranker` | `RAG_RERANKER` | `{reranker}` |
| HTTP bearer token | `--auth-token` | `RAG_AUTH_TOKEN` | none |
| embedding endpoint | `--embed-base-url` | `RAG_EMBED_BASE_URL` | none |
| embedding model | `--embed-model` | `RAG_EMBED_MODEL` | from the index |
| embedding API key | — | `RAG_EMBED_API_KEY` | none |
| config file | `--config PATH` | `RAG_CONFIG` | `./rag.config.json` |

`install.sh`, `run-http.sh`, `run-mcp.sh` and `setup.sh` all accept these
flags (`--config`, `--device`, `--host`, `--port`, `--top-k`, `--no-reranker`,
`--keyfile`); `install.sh` saves the ones you give it into `rag.config.json`.
Inspect the merged result any time (secrets are redacted):

```bash
.venv/bin/python server.py --print-config
```

**Network safety:** the server binds `127.0.0.1` by default and sends no CORS
headers. Binding any other address (`--host 0.0.0.0`) requires an auth token
(`RAG_AUTH_TOKEN` or `--auth-token`); without one it refuses to start. Clients
send `Authorization: Bearer <token>`.

{embed_section}

## Use it as an MCP server (agents)

- **Claude Desktop** — put this in `claude_desktop_config.json` (edit the path):

```
{mcp_snippet}
```

- **OpenClaw** — `mcp.servers` in `openclaw.json`, same command/args shape.
- **Cursor / other MCP clients** — stdio command: `bash run-mcp.sh`.
{mcp_enc_note}
Tools exposed: `rag_search(query, top_k)` and `rag_info()`.

## HTTP API

| Endpoint | Returns |
|---|---|
| `GET /health` | corpus name, chunk count, search mode |
| `GET /search?q=...&top_k=5` | ranked passages with source file names |
| `POST /search` (JSON `{{"query": "...", "top_k": 5}}`) | same |

## Notes

- Everything is local; the server makes no calls except to the embedding
  endpoint you configure (and none at all when the model is bundled).
- {sources_note}
- Re-export from Finetune Studio any time the documents change.
"""

_SECURITY_ENCRYPTED = """## Encrypted at rest

All content — chunks, original documents, vectors, the keyword index, document
names and metadata — lives in `corpus/corpus.enc`, an authenticated-encrypted
container (AES-256-GCM in 64 KiB frames, key derived from your passphrase with
scrypt). The only plaintext in it is a small header: format version, KDF
parameters, salt and nonce scheme.

- **The passphrase is not in this package.** It was shown once at export time.
  Without it nothing can be read; if you lose it the pack cannot be recovered.
- The server decrypts **in memory only**: no temp files, no extracted
  directories, no cache of chunk text.
- A wrong passphrase, a modified byte or a truncated file fails closed with an
  error — never wrong or partial answers.
- Give the passphrase to the server by (first match wins): `--keyfile PATH`,
  env `RAG_PASSPHRASE`, env `RAG_KEYFILE` (a path), or an interactive prompt.
  For MCP (stdio) and services use a keyfile (`chmod 600`) or the env var; a
  prompt only works from a terminal.
- **Not encrypted:** bundled model weights (`corpus/embedder/`,
  `corpus/reranker/`) are public models and ship as-is. The rest of this
  README, the scripts and `rag.config.json` contain no document content.

"""

_SECURITY_PLAIN = """## Security note — NOT ENCRYPTED

This pack was exported with encryption **disabled** (`--no-encrypt`). The
documents, chunks and index in `corpus/` are plain files: anyone who has the
archive can read everything. Re-export with encryption on for sensitive data.

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
automatically (`--no-reranker` turns it off).

`install.sh` sets up the venv for this (numpy + sentence-transformers; the
torch wheel is the big download, a few minutes once). The first search
loads the model into RAM (~1-2 GB) and takes a second or two; after that it
is fast. The device defaults to `auto`: the GPU (NVIDIA CUDA, AMD ROCm, Intel
XPU or Apple MPS) when one is usable, the CPU only otherwise. Force one with
`--device cpu|cuda|cuda:N|xpu|mps` (or `RAG_DEVICE=…`).
"""

_INSTALL_SH = r"""#!/usr/bin/env bash
# Creates a private venv and installs the dependencies (see requirements.txt).
# Optional runtime flags are saved into rag.config.json:
#   bash install.sh [--config PATH] [--device D] [--host H] [--port N]
#                   [--top-k N] [--no-reranker]
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
if [ -d corpus/embedder ]; then
  # GPU first: pick the torch wheel that matches the hardware; CPU wheel only when
  # no GPU is present. Override with TORCH_INDEX_URL (e.g. .../whl/cu128, .../whl/rocm6.4).
  if [ -n "${TORCH_INDEX_URL:-}" ]; then
    TORCH_ARGS=(--index-url "$TORCH_INDEX_URL"); KIND="custom index"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    TORCH_ARGS=(); KIND="NVIDIA CUDA"          # default PyPI Linux wheel bundles CUDA
  elif [ "$(uname -s)" = "Darwin" ]; then
    TORCH_ARGS=(); KIND="Apple MPS"            # default macOS wheel includes MPS
  elif command -v rocm-smi >/dev/null 2>&1 || [ -d /opt/rocm ]; then
    TORCH_ARGS=(--index-url https://download.pytorch.org/whl/rocm6.4); KIND="AMD ROCm"
  elif command -v xpu-smi >/dev/null 2>&1; then
    TORCH_ARGS=(--index-url https://download.pytorch.org/whl/xpu); KIND="Intel XPU"
  else
    TORCH_ARGS=(--index-url https://download.pytorch.org/whl/cpu); KIND="CPU (no GPU found)"
  fi
  echo "Model package: installing torch ($KIND) + sentence-transformers — a few minutes, once…"
  .venv/bin/pip install --quiet torch "${TORCH_ARGS[@]}"
fi
.venv/bin/pip install --quiet -r requirements.txt
if [ $# -gt 0 ]; then
  .venv/bin/python server.py --save-config "$@"
fi
echo "Installed. Next:  bash run-http.sh   or   bash run-mcp.sh   (flags pass through)"
"""

_RUN_HTTP_SH = r"""#!/usr/bin/env bash
# Starts the REST API. Flags pass through to server.py:
#   bash run-http.sh [--config PATH] [--host H] [--port N] [--top-k N]
#                    [--device D] [--no-reranker] [--keyfile PATH]
# Non-loopback --host needs RAG_AUTH_TOKEN (or --auth-token).
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || bash install.sh
[ -n "${PORT:-}" ] && export RAG_PORT="${RAG_PORT:-$PORT}"
exec .venv/bin/python server.py --http "$@"
"""

_RUN_MCP_SH = r"""#!/usr/bin/env bash
# Starts the MCP server on stdio. Flags pass through to server.py:
#   bash run-mcp.sh [--config PATH] [--top-k N] [--device D] [--no-reranker]
#                   [--keyfile PATH]
# Encrypted packs: stdio has no terminal prompt, so give the passphrase via
# --keyfile PATH, RAG_KEYFILE or RAG_PASSPHRASE.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || bash install.sh
exec .venv/bin/python server.py --mcp "$@"
"""

_SETUP_SH = r"""#!/usr/bin/env bash
# One-command setup for this Finetune Studio RAG pack.
#
#   bash setup.sh              guided: port, start server, optional service
#   bash setup.sh --yes        defaults only (install + MCP entry, no prompts)
#   bash setup.sh --yes --start --port 8899
#   bash setup.sh --service    run as a persistent systemd user service
#   bash setup.sh --uninstall  remove the service (keeps files)
#
# Pass-through flags (forwarded to server.py): --config PATH --device D
#   --host H --port N --top-k N --no-reranker --keyfile PATH
#
# Always writes MCP-ENTRY.txt with a copy-pasteable MCP config that has
# this machine's absolute paths already filled in.
set -euo pipefail
cd "$(dirname "$0")"
PKG_DIR="$(pwd)"
NAME="$(basename "$PKG_DIR")"
UNIT="fts-rag-$NAME.service"
UNIT_PATH="$HOME/.config/systemd/user/$UNIT"
PORT=""
EXTRA=()          # flags forwarded to server.py (never the port: handled below)
KEYFILE=""
ASSUME_YES=0; DO_START=0; DO_SERVICE=0; DO_UNINSTALL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y)      ASSUME_YES=1 ;;
    --port)        shift; PORT="${1:?--port needs a number}" ;;
    --port=*)      PORT="${1#--port=}" ;;
    --config|--device|--host|--top-k)
                   EXTRA+=("$1" "${2:?$1 needs a value}"); shift ;;
    --config=*|--device=*|--host=*|--top-k=*|--no-reranker) EXTRA+=("$1") ;;
    --keyfile)     shift; KEYFILE="$(cd "$(dirname "${1:?--keyfile needs a path}")" && pwd)/$(basename "$1")" ;;
    --start)       DO_START=1 ;;
    --service)     DO_SERVICE=1 ;;
    --uninstall)   DO_UNINSTALL=1 ;;
    -h|--help)     sed -n '2,15p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)"; exit 2 ;;
  esac
  shift
done
[ -n "$KEYFILE" ] && EXTRA+=(--keyfile "$KEYFILE")
ENCRYPTED=0; [ -f corpus/corpus.enc ] && ENCRYPTED=1

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

# 2) effective host/port (flags > env > rag.config.json > defaults), then
#    prompt for the port only when interactive and none was given.
PORT_ARGS=(); [ -n "$PORT" ] && PORT_ARGS=(--port "$PORT")
read -r EFF_HOST EFF_PORT < <(.venv/bin/python server.py --print-config "${EXTRA[@]}" "${PORT_ARGS[@]}" \
  | .venv/bin/python -c 'import json,sys; c=json.load(sys.stdin)["config"]; print(c["host"], c["port"])')
if [ -z "$PORT" ] && [ "$ASSUME_YES" = 0 ] && [ -t 0 ]; then
  read -r -p "Port for the search API [$EFF_PORT]: " ans || ans=""
  EFF_PORT="${ans:-$EFF_PORT}"
fi
case "$EFF_PORT" in (*[!0-9]*|'') EFF_PORT=8899 ;; esac
CURL_HOST="$EFF_HOST"; case "$EFF_HOST" in 0.0.0.0|::|'') CURL_HOST=127.0.0.1 ;; esac
SERVER_ARGS=("${EXTRA[@]}" --port "$EFF_PORT")

# 3) copy-paste MCP entry (absolute paths, ready to paste)
if [ "$ENCRYPTED" = 1 ] && [ -n "$KEYFILE" ]; then
  MCP_ARGS="\"$PKG_DIR/run-mcp.sh\", \"--keyfile\", \"$KEYFILE\""
else
  MCP_ARGS="\"$PKG_DIR/run-mcp.sh\""
fi
cat > MCP-ENTRY.txt <<ENTRY
{
  "mcpServers": {
    "$NAME": {
      "command": "bash",
      "args": [$MCP_ARGS]
    }
  }
}
ENTRY
echo
echo "=== Copy-paste MCP entry (also saved to MCP-ENTRY.txt) ==="
cat MCP-ENTRY.txt
echo "=== Paste into Claude Desktop / OpenClaw mcp.servers / Cursor; or use HTTP: ==="
echo "  curl 'http://$CURL_HOST:$EFF_PORT/search?q=your+question&top_k=5'"
if [ "$ENCRYPTED" = 1 ] && [ -z "$KEYFILE" ] && [ -z "${RAG_PASSPHRASE:-}" ] && [ -z "${RAG_KEYFILE:-}" ]; then
  echo "NOTE: this pack is encrypted. MCP/services need the passphrase non-interactively:"
  echo "  re-run with --keyfile /path/to/passphrase.txt (chmod 600), or set RAG_PASSPHRASE."
fi
echo

HAVE_SYSTEMD=0
if command -v systemctl >/dev/null 2>&1 && [ -n "${XDG_RUNTIME_DIR:-}" ]; then HAVE_SYSTEMD=1; fi
CAN_UNATTEND=1
if [ "$ENCRYPTED" = 1 ] && [ -z "$KEYFILE" ] && [ -z "${RAG_KEYFILE:-}" ]; then CAN_UNATTEND=0; fi

# 4) persistent service?
if [ "$DO_SERVICE" = 1 ] || { [ "$CAN_UNATTEND" = 1 ] && ask_yn "Install as a service that stays up (starts on login, restarts on crash)?" N; }; then
  if [ "$HAVE_SYSTEMD" = 0 ]; then
    echo "No systemd user session here — skipping service. Start manually with: bash run-http.sh"
  elif [ "$CAN_UNATTEND" = 0 ]; then
    echo "Encrypted pack: a service needs --keyfile PATH (a passphrase file) — skipping service."
  else
    ENV_LINES=""
    for v in RAG_EMBED_BASE_URL RAG_EMBED_API_KEY RAG_AUTH_TOKEN; do
      [ -n "${!v:-}" ] && ENV_LINES="$ENV_LINES
Environment=$v=${!v}"
    done
    mkdir -p "$(dirname "$UNIT_PATH")"
    ( umask 077; cat > "$UNIT_PATH" <<UNIT
[Unit]
Description=Finetune Studio RAG pack ($NAME)
After=network-online.target

[Service]
WorkingDirectory=$PKG_DIR
ExecStart=$PKG_DIR/.venv/bin/python $PKG_DIR/server.py --http ${SERVER_ARGS[*]}$ENV_LINES
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
UNIT
    )
    systemctl --user daemon-reload
    systemctl --user enable --now "$UNIT"
    echo "Service installed: $UNIT (port $EFF_PORT)"
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
if [ "$DO_START" = 1 ] || ask_yn "Start the search server now on port $EFF_PORT?" Y; then
  if [ "$ENCRYPTED" = 1 ] && [ -z "$KEYFILE" ] && [ -z "${RAG_PASSPHRASE:-}" ] && [ -z "${RAG_KEYFILE:-}" ]; then
    echo "Encrypted pack: start it from a terminal so you can type the passphrase:"
    echo "  bash run-http.sh ${EXTRA[*]:-} --port $EFF_PORT"
    exit 0
  fi
  nohup .venv/bin/python server.py --http "${SERVER_ARGS[@]}" > rag-server.log 2>&1 &
  echo $! > rag-server.pid
  sleep 2
  AUTH=(); [ -n "${RAG_AUTH_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $RAG_AUTH_TOKEN")
  if curl -fsS "${AUTH[@]}" "http://$CURL_HOST:$EFF_PORT/health" >/dev/null 2>&1; then
    echo "Search server running on http://$CURL_HOST:$EFF_PORT (log: rag-server.log, stop: kill \$(cat rag-server.pid))"
  else
    echo "Server started but /health not answering yet — check rag-server.log"
  fi
else
  echo "Not started. Whenever you like: bash run-http.sh  (flags: --port N --host H ...)"
fi
"""


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "corpus").lower()).strip("-")
    return s[:48] or "corpus"


_SKIP_PARTS = frozenset({"__pycache__", ".locks"})
_FMT_ALIASES = {"tgz": "tar.gz"}


class _Sink:
    """Streams members straight into the output archive (tar / tar.gz / zip).

    Nothing is staged on disk: bytes and file objects go from memory / the
    source files into the archive.  The archive itself is written to
    ``<out>.part`` and renamed on success.
    """

    def __init__(self, path: Path, fmt: str, *, fast: bool) -> None:
        self.fmt = fmt
        self._now = int(time.time())
        if fmt == "zip":
            self._zf = zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED,
                                       compresslevel=1 if fast else 9)
            self._tf = None
        elif fmt in ("tar.gz", "tar"):
            gz = fmt == "tar.gz"
            kwargs = {"compresslevel": 1 if fast else 9} if gz else {}
            # safetensors barely compress: level 1 for model packs.
            self._tf = tarfile.open(path, "w:gz" if gz else "w", **kwargs)  # noqa: SIM115
            self._zf = None
        else:
            raise ValueError(f"unsupported package format: {fmt!r}")

    def add_stream(self, arc: str, fp: BinaryIO, size: int, mode: int = 0o644) -> None:
        if self._tf is not None:
            ti = tarfile.TarInfo(arc)
            ti.size, ti.mode, ti.mtime = size, mode, self._now
            self._tf.addfile(ti, fp)
        else:
            zi = zipfile.ZipInfo(arc, date_time=time.localtime(self._now)[:6])
            zi.external_attr = (0o100000 | mode) << 16
            zi.compress_type = zipfile.ZIP_DEFLATED
            with self._zf.open(zi, "w", force_zip64=size >= 2 ** 31) as dst:
                shutil.copyfileobj(fp, dst, _COPY_BUF)

    def add_bytes(self, arc: str, data: bytes | str, mode: int = 0o644) -> None:
        import io
        raw = data.encode("utf-8") if isinstance(data, str) else data
        self.add_stream(arc, io.BytesIO(raw), len(raw), mode)

    def add_file(self, arc: str, path: Path, mode: int = 0o644) -> None:
        with open(path, "rb") as fp:
            self.add_stream(arc, fp, path.stat().st_size, mode)

    def close(self) -> None:
        (self._tf or self._zf).close()


def _resolve_shared(ref: str, kind: str) -> tuple[str, Path] | None:
    """``shared:<kind>:<short_id>`` -> (short_id, source dir); None otherwise."""
    prefix = f"shared:{kind}:"
    ref = str(ref or "")
    if not ref.startswith(prefix):
        return None
    from finetune_studio.data import shared_models as sm
    short_id = ref[len(prefix):]
    return short_id, sm.resolve(short_id, kind)


def _add_model_dir(sink: _Sink, src: Path, arc_dir: str) -> None:
    for f in sorted(src.rglob("*")):
        if not f.is_file() or f.is_symlink():
            continue
        rel = f.relative_to(src)
        if _SKIP_PARTS & set(rel.parts) or f.name.endswith(".lock"):
            continue
        sink.add_file(f"{arc_dir}/{rel.as_posix()}", f)


def _readme(*, title: str, slug: str, embed_model: str, encrypted: bool,
            models: bool, cfg: RagExportConfig) -> str:
    mcp_snippet = _mcp_config_example(slug if encrypted else f"{slug}-rag",
                                      NEUTRAL_ROOT if encrypted else f"{slug}-rag",
                                      encrypted).rstrip("\n")
    mcp_enc_note = ("\nThis pack is encrypted: the MCP client cannot type a "
                    "passphrase, so pass `--keyfile` (as above) or set "
                    "`RAG_PASSPHRASE` in the client's `env`.\n") if encrypted else ""
    embed_section = (_README_SECTION_OFFLINE if models else _README_SECTION_API)
    return _README_TMPL.format(
        title=title, date=time.strftime("%Y-%m-%d"),
        enc_note=" (**encrypted** in `corpus/corpus.enc`)" if encrypted else "",
        container_note=("; `rag_container.py` next to it opens the encrypted corpus"
                        if encrypted else ""),
        security_section=_SECURITY_ENCRYPTED if encrypted else _SECURITY_PLAIN,
        curl_hint='curl "http://127.0.0.1:8899/search?q=your+question&top_k=5"'
                  + ('   # add -H "Authorization: Bearer $TOKEN" if a token is set'),
        host=cfg.host, port=cfg.port, top_k=cfg.top_k, device=cfg.device,
        reranker="on" if cfg.reranker_enabled else "off",
        embed_section=embed_section.format(embed_model=embed_model),
        mcp_snippet=mcp_snippet, mcp_enc_note=mcp_enc_note,
        sources_note=("the original documents are inside `corpus.enc` "
                      "(read lazily, in memory)." if encrypted else
                      "`corpus/sources/` keeps the parsed text of every document."),
    )


def _mcp_config_example(slug: str, root: str, encrypted: bool) -> str:
    args = [f"/ABSOLUTE/PATH/TO/{root}/run-mcp.sh"]
    if encrypted:
        args += ["--keyfile", "/ABSOLUTE/PATH/TO/passphrase.txt"]
    return json.dumps({"mcpServers": {slug: {"command": "bash", "args": args}}},
                      indent=2) + "\n"


def _generate_passphrase() -> str:
    return secrets.token_urlsafe(18)  # 144 bits


def build_package(
    corpus_dir: str | Path,
    out_path: str | Path,
    *,
    name: str | None = None,
    fmt: str | None = None,
    include_models: bool | None = None,
    config: RagExportConfig | None = None,
    encrypt: bool | None = None,
    passphrase: str | None = None,
) -> PackageResult:
    """Assemble the hostable package for a built corpus and archive it.

    ``corpus_dir`` must contain a built PortableRAG corpus (manifest.json +
    chunks.parquet + vectors.npy + vectors.idx.json + bm25.json).

    ``config`` supplies the export defaults (format, models, device, host,
    port, top_k, reranker, encryption, KDF cost); the explicit ``fmt`` /
    ``include_models`` / ``encrypt`` arguments override it.  With encryption on
    (the default) ``passphrase`` is used, or a strong one is generated and
    returned once in the result.  The key is never written anywhere.
    """
    import pandas as pd  # local import: only needed at export time

    cfg = config or RagExportConfig()
    fmt = _FMT_ALIASES.get(fmt or cfg.archive_format, fmt or cfg.archive_format)
    if fmt not in ("tar.gz", "tar", "zip"):
        raise ValueError(f"unsupported package format: {fmt!r}")
    models = cfg.include_models if include_models is None else include_models
    encrypted = cfg.encrypt if encrypt is None else encrypt
    if passphrase is not None and encrypted and len(passphrase) < MIN_PASSPHRASE_LEN:
        raise ValueError(f"passphrase too short (minimum {MIN_PASSPHRASE_LEN} characters)")
    if passphrase is not None and not encrypted:
        raise ValueError("a passphrase was given but encryption is disabled")

    corpus = Path(corpus_dir)
    for req in ("manifest.json", "chunks.parquet", "vectors.npy",
                "vectors.idx.json", "bm25.json"):
        if not (corpus / req).is_file():
            raise FileNotFoundError(f"corpus missing {req}: {corpus}")
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    title = name or manifest.get("name") or corpus.name
    if encrypted:  # nothing derived from the project/document names in paths/README
        title, slug, root = "Portable RAG pack", NEUTRAL_ROOT, NEUTRAL_ROOT
    else:
        slug = _slug(title)
        root = f"{slug}-rag"

    # Resolve + validate models BEFORE any output is written.
    embed_model = str((manifest.get("embedding_model") or {}).get("name")
                      or "the embedding model named in corpus/manifest.json")
    emb_src = rr_src = None
    if models:
        rs = manifest.setdefault("rag_settings", {})
        emb = _resolve_shared((manifest.get("embedding_model") or {}).get("name", ""),
                              "embedder")
        if not emb:
            raise FileNotFoundError(
                "include_models requested but this corpus does not "
                "reference a shared embedder — rebuild it once from the "
                "RAG page, then export with models")
        emb_id, emb_src = emb
        rr = _resolve_shared(rs.get("reranker", ""), "reranker") \
            if cfg.include_reranker else None
        # Rewrite refs for display only — server.py loads bundled models by
        # directory, never by these paths.
        manifest["embedding_model"]["name"] = emb_id
        rs["embedder"] = emb_id
        if rr:
            rr_id, rr_src = rr
            rs["reranker"] = rr_id
        else:
            rs["rerank_enabled"] = False
        embed_model = emb_id

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    generated: str | None = None
    if encrypted and passphrase is None:
        generated = passphrase = _generate_passphrase()

    sink = _Sink(part, fmt, fast=models)
    spool = None
    ok = False
    try:
        # ---- corpus ----------------------------------------------------
        chunks_jsonl = pd.read_parquet(corpus / "chunks.parquet").to_json(
            orient="records", lines=True, force_ascii=False)
        manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
        source_files = []
        if (corpus / "sources").is_dir():
            source_files = [f for f in sorted((corpus / "sources").rglob("*"))
                            if f.is_file() and not f.is_symlink()]
        if encrypted:
            from .rag_container import ContainerWriter
            spool = tempfile.TemporaryFile()  # noqa: SIM115 - anonymous, ciphertext only
            cw = ContainerWriter(spool, passphrase, log_n=cfg.kdf_log_n)
            cw.add("manifest.json", manifest_json.encode("utf-8"))
            cw.add("chunks.jsonl", chunks_jsonl.encode("utf-8"))
            for fname in ("vectors.npy", "vectors.idx.json", "bm25.json"):
                with open(corpus / fname, "rb") as fp:
                    cw.add(fname, fp)
            for f in source_files:
                with open(f, "rb") as fp:
                    cw.add(f"sources/{f.relative_to(corpus / 'sources').as_posix()}", fp)
            cw.close()
            size = spool.tell()
            spool.seek(0)
            sink.add_stream(f"{root}/corpus/{ENC_NAME}", spool, size)
        else:
            sink.add_bytes(f"{root}/corpus/manifest.json", manifest_json)
            sink.add_bytes(f"{root}/corpus/chunks.jsonl", chunks_jsonl)
            for fname in ("vectors.npy", "vectors.idx.json", "bm25.json"):
                sink.add_file(f"{root}/corpus/{fname}", corpus / fname)
            for f in source_files:
                sink.add_file(
                    f"{root}/corpus/sources/{f.relative_to(corpus / 'sources').as_posix()}", f)
        del chunks_jsonl, manifest_json

        # ---- models (public weights; streamed, never staged) -----------
        if emb_src:
            _add_model_dir(sink, emb_src, f"{root}/corpus/embedder")
        if rr_src:
            _add_model_dir(sink, rr_src, f"{root}/corpus/reranker")

        # ---- runtime files ---------------------------------------------
        requirements = "numpy\n" + ("cryptography>=42\n" if encrypted else "")
        if models:
            requirements += "sentence-transformers>=3.0\n"
        sink.add_bytes(f"{root}/server.py", _SERVER_SRC.read_text(encoding="utf-8"))
        if encrypted:
            sink.add_bytes(f"{root}/rag_container.py",
                           _CONTAINER_SRC.read_text(encoding="utf-8"))
        sink.add_bytes(f"{root}/requirements.txt", requirements)
        sink.add_bytes(f"{root}/{RUNTIME_CONFIG_NAME}",
                       json.dumps(cfg.runtime_config(), indent=2) + "\n")
        sink.add_bytes(f"{root}/mcp-config.example.json",
                       _mcp_config_example(f"{slug}" if encrypted else f"{slug}-rag",
                                           root, encrypted))
        for sh, body in (("install.sh", _INSTALL_SH), ("run-http.sh", _RUN_HTTP_SH),
                         ("run-mcp.sh", _RUN_MCP_SH), ("setup.sh", _SETUP_SH)):
            sink.add_bytes(f"{root}/{sh}", body, 0o755)
        sink.add_bytes(f"{root}/README.md", _readme(
            title=title, slug=slug, embed_model=embed_model,
            encrypted=encrypted, models=models, cfg=cfg))
        sink.close()
        part.replace(out)
        ok = True
    finally:
        if spool is not None:
            spool.close()
        if not ok:
            with contextlib.suppress(Exception):  # best-effort cleanup of a failed build
                sink.close()
            part.unlink(missing_ok=True)

    log.info("rag package built: %s (%d bytes, encrypted=%s)", out, out.stat().st_size, encrypted)
    return PackageResult(path=out, encrypted=encrypted, root=root,
                         size=out.stat().st_size, passphrase=generated)
