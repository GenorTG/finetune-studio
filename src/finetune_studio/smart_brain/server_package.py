"""Build a hostable, self-installing RAG package from a BrainIndex corpus.

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
    ├── brain_container.py           encrypted-container reader (encrypted packs)
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
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from .export_config import RUNTIME_CONFIG_NAME, BrainExportConfig

log = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_SERVER_SRC = _HERE / "standalone_server.py"
_CONTAINER_SRC = _HERE / "brain_container.py"
ENC_NAME = "corpus.enc"
ORIGINALS_INDEX = "originals.json"
ORIGINALS_DIR = "originals"
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
    layers: dict[str, bool] = field(default_factory=dict)   # what the package can run (manifest "search_layers")
    runtime_search: str = ""      # how it finds passages where it runs (BrainExportConfig.runtime_search)


_README_TMPL = """# {title}

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
| `update.sh` | replace this package with a newer version of it, keeping your config (`setup.sh --update`) |

{security_section}{layers_section}## Quickstart

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
open http://127.0.0.1:8899/   # browser chat and provider connection
{curl_hint}
```

The browser chat can connect directly to OpenAI or Anthropic Claude (or an
OpenAI-compatible endpoint). Enter the provider URL, model and API key in the
page; the key is sent for that request only and is not written into the package
or its config. For a network-hosted instance, put it behind HTTPS and set
`RAG_AUTH_TOKEN`; enter that service token in the page as well. Provider usage
is billed to the account whose key you enter.

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
| meaning search | `--semantic` / `--no-semantic` | `RAG_SEMANTIC` | `{semantic}` |
| word search | `--keyword` / `--no-keyword` | `RAG_KEYWORD` | `{keyword}` |
| answer generation | `--answer` / `--no-answer` | `RAG_ANSWER` | `{answer}` |
| answer model endpoint | `--llm-base-url` | `RAG_LLM_BASE_URL` | `{llm_base_url}` |
| answer model | `--llm-model` | `RAG_LLM_MODEL` | `{llm_model}` |
| answer model API key | — | `RAG_LLM_API_KEY` | none |
| follow-up searches per answer (0 = off) | — | `RAG_FOLLOW_UP` | `1` |
| HTTP bearer token | `--auth-token` | `RAG_AUTH_TOKEN` | none |
| embedding endpoint | `--embed-base-url` | `RAG_EMBED_BASE_URL` | none |
| embedding model | `--embed-model` | `RAG_EMBED_MODEL` | from the index |
| embedding API key | — | `RAG_EMBED_API_KEY` | none |
| config file | `--config PATH` | `RAG_CONFIG` | `./rag.config.json` |

`install.sh`, `run-http.sh`, `run-mcp.sh` and `setup.sh` all accept these
flags (`--config`, `--device`, `--host`, `--port`, `--top-k`, `--no-reranker`,
`--no-semantic`, `--no-keyword`, `--answer`, `--no-answer`, `--keyfile`); `install.sh` saves the ones you give it into `rag.config.json`.
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
- **Claude Code** — `claude mcp add {slug} -- bash /ABSOLUTE/PATH/run-mcp.sh{kf_cli}` (stdio), or over HTTP below.
- **Codex** — `[mcp_servers.{slug}]` in `~/.codex/config.toml` with `command = "bash"`, `args = ["/ABSOLUTE/PATH/run-mcp.sh"{kf_toml}]`.
- **Cursor / other MCP clients** — stdio command: `bash run-mcp.sh{kf_cli}`.
{mcp_enc_note}
Tools exposed: {tools}. Clients that read server instructions also get the package's identity (who it speaks for,
how to answer) up front; any client can read it with `rag_profile`.

### MCP over HTTP (remote clients, web apps)

`bash run-http.sh` also serves MCP at `POST /mcp` (Streamable HTTP; clients from the 2024-11-05 handshake up to the
stateless 2026-07-28 revision). On this machine:

```bash
claude mcp add --transport http {slug} http://127.0.0.1:{port}/mcp
```

From another machine, start it with a token (`RAG_AUTH_TOKEN`, required off loopback), put it behind HTTPS and send
`Authorization: Bearer <token>`, e.g. `claude mcp add --transport http {slug} https://your.host/mcp --header
"Authorization: Bearer $TOKEN"`. Browser pages from other sites are refused (Origin check).

## Chat page

`bash run-http.sh` serves a chat page at `http://127.0.0.1:{port}/`: ask the documents, read the passages each
answer came from. It answers with this package's own connection when one is set (below), or with a provider picked in
the page: OpenAI, Anthropic Claude, OpenRouter, a local server (Ollama, LM Studio, llama.cpp) or any OpenAI-compatible
API. A key typed there is sent with each question and never saved by the server.

## HTTP API

| Endpoint | Returns |
|---|---|
| `GET /health` | corpus name, chunk count, search mode |
| `GET /search?q=...&top_k=5` | ranked passages with source file names |
| `POST /search` (JSON `{{"query": "...", "top_k": 5}}`) | same |
| `GET /ask?q=...` / `POST /ask` (JSON `{{"question": "..."}}`) | the LLM's answer + sources (answer generation on) |

## Notes

- Everything is local; the server makes no calls except to the embedding
  and answer endpoints you configure (none at all with the embedder bundled and answers off).
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

The index was built with **{embed_model}**. Meaning search needs that model to
turn each question into a vector: without it (or an endpoint that serves it) only
word search (BM25) runs, when this package has it.
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
loads the models into memory (~1-4 GB) and takes a few seconds; after that it
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
if [ -d corpus/embedder ] || [ -d corpus/reranker ]; then
  # GPU first: pick the torch wheel that matches the hardware; CPU wheel only when
  # no GPU is present. Override with TORCH_INDEX_URL (e.g. .../whl/cu128, .../whl/rocm6.4).
  if [ -n "${TORCH_INDEX_URL:-}" ]; then
    TORCH_ARGS=(--index-url "$TORCH_INDEX_URL"); KIND="custom index"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    # The default PyPI wheel (cu13x) has kernels for sm_75+ only: on an older card (Pascal, Volta)
    # it installs fine and then cannot run, so those get the cu126 build (sm_50+).
    MIN_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -n | head -1)
    if [ -n "$MIN_CC" ] && awk -v cc="$MIN_CC" 'BEGIN { exit !(cc + 0 < 7.5) }'; then
      TORCH_ARGS=(--index-url https://download.pytorch.org/whl/cu126); KIND="NVIDIA CUDA (sm_${MIN_CC/./}: cu126 build)"
    else
      TORCH_ARGS=(); KIND="NVIDIA CUDA"        # default PyPI Linux wheel bundles CUDA
    fi
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

_UPDATE_SH = r"""#!/usr/bin/env bash
# Replace this package with a newer version of the same package, in place:
#   bash update.sh /path/to/new-version.tar.gz [--keyfile PATH] [--allow-widen] [--force]
# The corpus, server and scripts are replaced; rag.config.json takes the new version's settings and keeps the
# values you changed after install (it prints which). The venv and your keyfile stay. An encrypted version is
# opened first with --keyfile (default ./passphrase.txt) or RAG_PASSPHRASE, so a wrong passphrase changes nothing.
# Refused unless you add a flag: another package id or a version not newer (--force); a version that opens up more
# than this one, e.g. more documents, original files or file names shown (--allow-widen). A running service
# (setup.sh --service) is restarted.
set -euo pipefail
[ $# -ge 1 ] || { sed -n '2,10p' "$0" | sed 's/^# \?//'; exit 2; }
ARGS=()   # files given relative to where you are (the archive, the keyfile) still work after the cd below
for a in "$@"; do
  if [ -f "$a" ]; then ARGS+=("$(cd "$(dirname "$a")" && pwd)/$(basename "$a")"); else ARGS+=("$a"); fi
done
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || bash install.sh
.venv/bin/python server.py --update "${ARGS[@]}"
.venv/bin/pip install --quiet -r requirements.txt
UNIT="fts-brain-$(basename "$(pwd)").service"
if command -v systemctl >/dev/null 2>&1 && systemctl --user is-active --quiet "$UNIT" 2>/dev/null; then
  systemctl --user restart "$UNIT" && echo "Restarted $UNIT with the new version."
else
  echo "Restart the server (bash run-http.sh / run-mcp.sh, or your MCP client) to use the new version."
fi
"""

_SETUP_SH = r"""#!/usr/bin/env bash
# One-command setup for this Finetune Studio RAG pack.
#
#   bash setup.sh              guided: port, start server, optional service
#   bash setup.sh --yes        defaults only (install + MCP entry, no prompts)
#   bash setup.sh --yes --start --port 8899
#   bash setup.sh --service    run as a persistent systemd user service
#   bash setup.sh --uninstall  remove the service (keeps files)
#   bash setup.sh --update NEW.tar.gz [--keyfile PATH] [--force]
#                              replace this package with its newer version
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
UNIT="fts-brain-$NAME.service"
UNIT_PATH="$HOME/.config/systemd/user/$UNIT"
PORT=""
EXTRA=()          # flags forwarded to server.py (never the port: handled below)
KEYFILE=""
ASSUME_YES=0; DO_START=0; DO_SERVICE=0; DO_UNINSTALL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --update)      shift; exec bash "$PKG_DIR/update.sh" "$@" ;;
    --yes|-y)      ASSUME_YES=1 ;;
    --port)        shift; PORT="${1:?--port needs a number}" ;;
    --port=*)      PORT="${1#--port=}" ;;
    --config|--device|--host|--top-k)
                   EXTRA+=("$1" "${2:?$1 needs a value}"); shift ;;
    --llm-base-url|--llm-model)
                   EXTRA+=("$1" "${2:?$1 needs a value}"); shift ;;
    --config=*|--device=*|--host=*|--top-k=*|--no-reranker|--no-semantic|--no-keyword|--answer|--no-answer|--llm-base-url=*|--llm-model=*) EXTRA+=("$1") ;;
    --keyfile)     shift; KEYFILE="$(cd "$(dirname "${1:?--keyfile needs a path}")" && pwd)/$(basename "$1")" ;;
    --start)       DO_START=1 ;;
    --service)     DO_SERVICE=1 ;;
    --uninstall)   DO_UNINSTALL=1 ;;
    -h|--help)     sed -n '2,17p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)"; exit 2 ;;
  esac
  shift
done
ENCRYPTED=0; [ -f corpus/corpus.enc ] && ENCRYPTED=1
# Encrypted: MCP clients and services cannot type a passphrase. Use --keyfile, else ./passphrase.txt, else ask once
# (from a terminal) and write it there, readable only by you.
if [ "$ENCRYPTED" = 1 ] && [ -z "$KEYFILE" ] && [ "$DO_UNINSTALL" = 0 ]; then
  if [ -f "$PKG_DIR/passphrase.txt" ]; then
    KEYFILE="$PKG_DIR/passphrase.txt"
  elif [ -z "${RAG_PASSPHRASE:-}" ] && [ -z "${RAG_KEYFILE:-}" ] && [ "$ASSUME_YES" = 0 ] && [ -t 0 ]; then
    read -r -s -p "Package passphrase (saved to passphrase.txt, chmod 600; Enter to skip): " PW || PW=""
    echo
    if [ -n "$PW" ]; then
      ( umask 077; printf '%s\n' "$PW" > "$PKG_DIR/passphrase.txt" )
      KEYFILE="$PKG_DIR/passphrase.txt"
    fi
    unset PW
  fi
fi
[ -n "$KEYFILE" ] && EXTRA+=(--keyfile "$KEYFILE")

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
if [ "$ENCRYPTED" = 1 ]; then
  MCP_ARGS="\"$PKG_DIR/run-mcp.sh\", \"--keyfile\", \"${KEYFILE:-$PKG_DIR/passphrase.txt}\""
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
  echo "NOTE: this pack is encrypted. The MCP entry above reads $PKG_DIR/passphrase.txt: write the passphrase"
  echo "  there (chmod 600), re-run with --keyfile /path/to/passphrase.txt, or set RAG_PASSPHRASE."
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
    for v in RAG_EMBED_BASE_URL RAG_EMBED_API_KEY RAG_AUTH_TOKEN RAG_LLM_BASE_URL RAG_LLM_MODEL RAG_LLM_API_KEY; do
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


def _resolve_or_register(ref: str, kind: str) -> tuple[str, Path] | None:
    """Like ``_resolve_shared``, but a model named by its Hugging Face id (the reranker the Smart Brain page's
    search settings save) is put in the shared store first. It used to be skipped: the package shipped with the
    reranker off and said nothing. A model that cannot be fetched is an error, not a silently smaller package."""
    found = _resolve_shared(ref, kind)
    if found or not ref or ref.startswith(("shared:", "embedder_local:", "reranker_local:")):
        return found
    from finetune_studio.data import shared_models as sm
    try:
        r = sm.register(ref, kind=kind, prefer_cached=True)
    except Exception as e:  # download / copy errors vary; the message is what the user needs
        raise FileNotFoundError(f"the {kind} {ref!r} could not be bundled: {e}") from e
    return r.short_id, Path(r.path)


def _add_model_dir(sink: _Sink, src: Path, arc_dir: str) -> None:
    for f in sorted(src.rglob("*")):
        if not f.is_file() or f.is_symlink():
            continue
        rel = f.relative_to(src)
        if _SKIP_PARTS & set(rel.parts) or f.name.endswith(".lock"):
            continue
        sink.add_file(f"{arc_dir}/{rel.as_posix()}", f)


def _layers_section(cfg: BrainExportConfig, *, bundled_embedder: bool, bundled_reranker: bool) -> str:
    def state(on: bool, note: str = "") -> str:
        return ("**on**" + (f" ({note})" if note else "")) if on else "off"
    sem_note = ("embedder bundled" if bundled_embedder else
                "needs `RAG_EMBED_BASE_URL`; without it the server uses word search only")
    rows = [
        ("Meaning search", "vectors: finds passages that mean the same in other words", "--no-semantic", "RAG_SEMANTIC",
         state(cfg.semantic_search, sem_note if cfg.semantic_search else "")),
        ("Word search", "BM25 keyword index: exact names, codes, numbers", "--no-keyword", "RAG_KEYWORD",
         state(cfg.keyword_search)),
        ("Rank fusion", "Reciprocal Rank Fusion merges the two rankings", "—", "—",
         state(cfg.semantic_search and cfg.keyword_search, "automatic when both searches are on")),
        ("Reranker", "a cross-encoder rescores the best candidates", "--reranker / --no-reranker", "RAG_RERANKER",
         state(cfg.reranker_enabled and bundled_reranker,
               "reranker bundled") if bundled_reranker or not cfg.reranker_enabled
         else "off (no reranker model in this package)"),
        ("Answer generation", "the G in RAG: an LLM writes the answer from the passages", "--answer / --no-answer",
         "RAG_ANSWER", state(cfg.answer_enabled, f"model `{cfg.llm_model or 'set RAG_LLM_MODEL'}`")),
    ]
    table = "\n".join(f"| {a} | {b} | `{c}` | `{d}` | {e} |" if c != "—" else f"| {a} | {b} | — | — | {e} |"
                      for a, b, c, d, e in rows)
    return f"""## Search layers

Every layer of this search pipeline is separate and can be switched off. The package only contains the files the
layers it was exported with need (no vectors without meaning search, no BM25 index without word search); a flag
can switch a shipped layer off, never on one that is not in the package.

**Pipeline:** {pipeline_text(cfg, bundled_reranker=bundled_reranker)}

| Layer | What it does | Flag | Env | In this package |
|---|---|---|---|---|
{table}

`GET /health` (and the MCP tool `rag_info`) reports which layers are actually running.

### Answer generation

Without it the server returns passages and your own model or agent writes the answer (the usual MCP setup). With it
on, `GET /ask?q=...` / `POST /ask` (and the MCP tool `rag_answer`) send the passages to an OpenAI-compatible chat
endpoint (Ollama, LM Studio, llama.cpp server, vLLM, OpenAI…) with the same grounded prompt as the studio's Smart
Brain chat, and return the answer with its sources. A question that reaches its answer through another item ("who
manages the site where asset X is installed?") finds the first item but not the second, which it never names: the
answer model may then ask for one more search (`SEARCH: …`) before it answers, and the reply lists it under
`follow_up_searches`. When the answer says part of the question is not stated, the server searches once more for
that part and answers again; the reply's `followup` names the search and the sources it added (`RAG_RETRY_MISSING=0`
turns it off). `RAG_FOLLOW_UP=0` turns both off; up to 3 follow-up searches are allowed.
File names in answers: {"**hidden** (a public bot: no source file is named; `/ask` returns `sources` only with `\"sources\": true`, `rag_answer` only with `include_sources`)" if cfg.hide_file_names else "shown (answers cite `[file name]`)"}; `RAG_HIDE_FILE_NAMES=1/0` changes it.

```bash
export RAG_LLM_PROVIDER="{cfg.llm_provider}"   # openai (any OpenAI-compatible API) or anthropic (Claude)
export RAG_LLM_BASE_URL="{cfg.llm_base_url or 'http://127.0.0.1:11434/v1'}"   # set at export: {cfg.llm_base_url or 'not set'}
export RAG_LLM_MODEL="{cfg.llm_model or 'your-model'}"
# export RAG_LLM_API_KEY="***"   # only for endpoints that need one; never stored in the package
bash run-http.sh --answer
curl "http://127.0.0.1:8899/ask?q=your+question"
```

"""


def _base_id(document_id: str) -> str:
    return str(document_id).split("~", 1)[0]


def _subset(corpus: Path, chunks_df, layer_files: tuple[str, ...], keep: set[str], manifest: dict):
    """The kept documents' corpus, in memory: their passages, their vectors (rows sliced, ids renumbered), a BM25
    index recomputed over their passages alone (document frequencies, lengths and averages of the subset), and the
    manifest's counts and document list cut to match. Returns ``(chunks_df, {file: bytes})``."""
    import io

    import numpy as np

    from .bm25 import BM25Index

    mask = chunks_df["document_id"].map(lambda d: _base_id(d) in keep)
    sub = chunks_df[mask].reset_index(drop=True)
    if sub.empty:
        raise ValueError("no passage of the chosen documents is in the index")
    data: dict[str, bytes | Path] = {}
    if "vectors.npy" in layer_files:
        vectors = np.load(corpus / "vectors.npy", allow_pickle=False)
        idx = json.loads((corpus / "vectors.idx.json").read_text(encoding="utf-8"))
        rows = [int(idx[cid]) for cid in sub["id"]]
        buf = io.BytesIO()
        np.save(buf, vectors[rows])
        data["vectors.npy"] = buf.getvalue()
        data["vectors.idx.json"] = json.dumps({cid: i for i, cid in enumerate(sub["id"])}).encode("utf-8")
    if "bm25.json" in layer_files:
        old = json.loads((corpus / "bm25.json").read_text(encoding="utf-8"))
        bm25 = BM25Index.build([str(t) for t in sub["text"]])
        bm25.k1, bm25.b = float(old.get("k1", bm25.k1)), float(old.get("b", bm25.b))
        data["bm25.json"] = json.dumps(bm25.to_dict()).encode("utf-8")
    extra = manifest.setdefault("extra", {})
    extra["documents_meta"] = [m for m in extra.get("documents_meta") or []
                               if _base_id(m.get("document_id") or "") in keep]
    if isinstance(extra.get("built_from"), dict):
        extra["built_from"] = {k: v for k, v in extra["built_from"].items() if _base_id(k) in keep}
    manifest["documents"] = len({_base_id(d) for d in sub["document_id"]})
    manifest["chunks"] = len(sub)
    return sub, data


def _originals_index(rows: list) -> bytes:
    return json.dumps([o.entry() for o in rows], ensure_ascii=False, indent=1).encode("utf-8")


_README_ORIGINALS = """## Original files

This package holds the {count} original file(s) the documents were read from{where}. **Anyone who can reach this
server can download them**: over MCP with `rag_get_document(document_id | filename)` (an embedded resource; files over
`file_inline_max_mb`, default 20 MB, come back as a link) and over HTTP at `GET /files/<document_id>` (same access
token as the other routes). `rag_list_documents` lists every document, and search hits carry `document_id` and
`original: true`. The chat page shows a download link under each source. Do not serve this package as a public bot
unless every file in it may be public.

"""


def _version_section(package: dict | None, encrypted: bool) -> str:
    """Version, id, date, parent; the document list only in a plaintext package (an encrypted README names none)."""
    if not package:
        return ""
    docs = package.get("documents") or []
    text = (f"## Version\n\nPackage `{package['package_id']}`, version **{package['version']}**, built "
            f"{package['created']}" + (f", replaces version {package['parent_version']}" if package.get(
                "parent_version") else " (first version)") + f". {len(docs)} document(s); `CHANGES.md` lists what "
            "changed. `rag_info` and `GET /health` report the same, with every document's name and sha256.\n\n"
            "Update a running copy in place: `bash update.sh /path/to/new-version.tar.gz"
            + (" --keyfile /path/to/passphrase.txt" if encrypted else "") + "` (the keyfile defaults to "
            "`passphrase.txt` beside `server.py`). It keeps the `rag.config.json` values you changed and the "
            "keyfile, takes the new version's other settings, and refuses a version that opens up more than the "
            "installed one (more documents, original files, file names shown) unless you add `--allow-widen`.\n\n")
    if not encrypted and docs:
        text += "| Document | sha256 | Size |\n|---|---|---|\n" + "".join(
            f"| {d['path']} | `{d['sha256'][:12]}` | {d['size']} |\n" for d in docs) + "\n"
    return text


def _scope_section(scope: dict | None) -> str:
    """Which documents are in the package. Folder/tag names only: no file names (an encrypted package's README
    holds no document names)."""
    if not scope:
        return ""
    n, total = scope.get("documents"), scope.get("of")
    if scope.get("label", "all documents") == "all documents":
        return f"## Documents in this package\n\nAll {n} documents of the index.\n\n"
    return (f"## Documents in this package\n\nA chosen part of the index: **{n} of {total} documents** "
            f"({scope.get('label')}). Only their passages, vectors, word index and files are inside; nothing else "
            "from the index can be found or read through this server (`rag_info` reports the same scope).\n\n")


def _readme(*, title: str, slug: str, embed_model: str, encrypted: bool,
            models: bool, cfg: BrainExportConfig, bundled_embedder: bool = False,
            bundled_reranker: bool = False, originals: int = 0, scope: dict | None = None,
            package: dict | None = None) -> str:
    mcp_snippet = _mcp_config_example(slug if encrypted else f"{slug}-brain",
                                      NEUTRAL_ROOT if encrypted else f"{slug}-brain",
                                      encrypted).rstrip("\n")
    mcp_enc_note = ("\nThis pack is encrypted: the MCP client cannot type a "
                    "passphrase, so pass `--keyfile` (as above) or set "
                    "`RAG_PASSPHRASE` in the client's `env`. `bash setup.sh` asks for the passphrase once and "
                    "writes it to `passphrase.txt` here (`chmod 600`); `run-mcp.sh`, `run-http.sh` and "
                    "`update.sh` use that file when no `--keyfile` is given.\n") if encrypted else ""
    embed_section = (_README_SECTION_OFFLINE if bundled_embedder else
                     _README_SECTION_API if cfg.semantic_search else "")
    return _README_TMPL.format(
        title=title, slug=slug, date=time.strftime("%Y-%m-%d"),
        enc_note=" (**encrypted** in `corpus/corpus.enc`)" if encrypted else "",
        container_note=("; `brain_container.py` next to it opens the encrypted corpus"
                        if encrypted else ""),
        security_section=_version_section(package, encrypted) + _scope_section(scope) + (_SECURITY_ENCRYPTED if encrypted else _SECURITY_PLAIN) + (
            _README_ORIGINALS.format(
            count=originals, where=" (inside the encrypted container)" if encrypted else " (`corpus/originals/`)")
            if originals else ""),
        curl_hint='curl "http://127.0.0.1:8899/search?q=your+question&top_k=5"'
                  + ('   # add -H "Authorization: Bearer $TOKEN" if a token is set'),
        host=cfg.host, port=cfg.port, top_k=cfg.top_k, device=cfg.device,
        reranker="on" if cfg.reranker_enabled and bundled_reranker else "off",
        semantic="on" if cfg.semantic_search else "off", keyword="on" if cfg.keyword_search else "off",
        answer="on" if cfg.answer_enabled else "off", llm_base_url=cfg.llm_base_url or "none",
        llm_model=cfg.llm_model or "none",
        embed_section=embed_section.format(embed_model=embed_model),
        mcp_snippet=mcp_snippet, mcp_enc_note=mcp_enc_note,
        kf_cli=" --keyfile /ABSOLUTE/PATH/passphrase.txt" if encrypted else "",
        kf_toml=', "--keyfile", "/ABSOLUTE/PATH/passphrase.txt"' if encrypted else "",
        layers_section=_layers_section(cfg, bundled_embedder=bundled_embedder, bundled_reranker=bundled_reranker),
        tools=", ".join(["`rag_search(query, top_k)`", "`rag_profile()`", "`rag_info()`", "`rag_list_documents()`"]
                        + (["`rag_get_document(document_id)`"] if originals else [])
                        + (["`rag_answer(question)`"] if cfg.answer_enabled else [])),
        sources_note=(("the parsed text" + (" and the original files" if originals else "") + " of every document "
                       "are inside `corpus.enc` (read lazily, in memory)." + ("" if originals else
                       " The original files are not in this package."))
                      if encrypted else
                      "`corpus/sources/` keeps the parsed text of every document."
                      + ("" if originals else " The original files are not in this package.")),
    )


def _mcp_config_example(slug: str, root: str, encrypted: bool) -> str:
    args = [f"/ABSOLUTE/PATH/TO/{root}/run-mcp.sh"]
    if encrypted:
        args += ["--keyfile", "/ABSOLUTE/PATH/TO/passphrase.txt"]
    return json.dumps({"mcpServers": {slug: {"command": "bash", "args": args}}},
                      indent=2) + "\n"


def layer_corpus_files(cfg: BrainExportConfig) -> tuple[str, ...]:
    """Index files the turned-on search layers need (meaning search: vectors; word search: the BM25 index)."""
    return ((("vectors.npy", "vectors.idx.json") if cfg.semantic_search else ())
            + (("bm25.json",) if cfg.keyword_search else ()))


def search_layers(cfg: BrainExportConfig) -> dict[str, bool]:
    """The layers chosen at export (``build_package`` records them in the manifest with ``reranker`` true only
    when a reranker model was bundled; the server decides from the files it finds)."""
    return {"semantic": cfg.semantic_search, "keyword": cfg.keyword_search,
            "fusion": cfg.semantic_search and cfg.keyword_search,
            "reranker": cfg.reranker_enabled, "answer": cfg.answer_enabled}


def pipeline_text(cfg: BrainExportConfig, *, bundled_reranker: bool) -> str:
    """One line: how a question travels through this package."""
    found = [n for n, on in (("meaning search (vectors)", cfg.semantic_search),
                             ("word search (BM25)", cfg.keyword_search)) if on]
    steps = ["question", " + ".join(found)]
    if len(found) == 2:
        steps.append("rank fusion (RRF)")
    if cfg.reranker_enabled and bundled_reranker:
        steps.append("reranker (cross-encoder)")
    steps.append("LLM answer with sources" if cfg.answer_enabled else "passages with sources")
    return " → ".join(steps)


def _generate_passphrase() -> str:
    return secrets.token_urlsafe(18)  # 144 bits


def build_package(
    corpus_dir: str | Path,
    out_path: str | Path,
    *,
    name: str | None = None,
    fmt: str | None = None,
    include_models: bool | None = None,
    config: BrainExportConfig | None = None,
    encrypt: bool | None = None,
    passphrase: str | None = None,
    identity: dict | None = None,
    originals: list | None = None,
    documents: set[str] | None = None,
    scope: dict | None = None,
    package: dict | None = None,
    changes_md: str = "",
    changes_md_public: str = "",
) -> PackageResult:
    """Assemble the hostable package for a built corpus and archive it.

    ``corpus_dir`` must contain a built BrainIndex corpus (manifest.json +
    chunks.parquet + vectors.npy + vectors.idx.json + bm25.json).

    ``config`` supplies the export defaults (format, models, device, host,
    port, top_k, reranker, encryption, KDF cost); the explicit ``fmt`` /
    ``include_models`` / ``encrypt`` arguments override it.  With encryption on
    (the default) ``passphrase`` is used, or a strong one is generated and
    returned once in the result.  The key is never written anywhere.

    ``originals`` (``smart_brain.originals.collect``) are the uploaded files; they ship only with
    ``config.include_originals``, inside the encrypted container when the package is encrypted.

    ``documents`` (``smart_brain.scope.resolve``) keeps only those document ids: the passages, their vectors, a BM25
    index recomputed over them, their parsed sources and originals; ``scope`` describes the choice (manifest,
    README, ``rag_info``). None = every document.

    ``package`` (``package_meta.make_record``) is the version record: inside the manifest with its document list,
    as a name-free ``package.json`` beside the corpus, and ``CHANGES.md`` (``changes_md``; an encrypted package gets
    ``changes_md_public``, without document names, in plaintext and the full one inside the container).
    """
    import pandas as pd  # local import: only needed at export time

    cfg = config or BrainExportConfig()
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
    # Each search layer ships only its own files: no vectors without meaning search, no BM25 index without word
    # search (the package is smaller and holds nothing it will not use).
    layer_files = layer_corpus_files(cfg)
    for req in ("manifest.json", "chunks.parquet", *layer_files):
        if not (corpus / req).is_file():
            raise FileNotFoundError(f"corpus missing {req}: {corpus}")
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    manifest["search_layers"] = search_layers(cfg)
    if identity:   # inside the manifest, so an encrypted package keeps it encrypted
        manifest["identity"] = dict(identity)
    keep = set(documents) if documents is not None else None
    if package:
        manifest["package"] = dict(package)
    ship_originals = [o for o in (originals or []) if keep is None or o.document_id in keep] \
        if cfg.include_originals else []
    manifest["original_files"] = len(ship_originals)
    title = name or manifest.get("name") or corpus.name
    if encrypted:  # nothing derived from the project/document names in paths/README
        title, slug, root = "Smart Brain knowledge package", NEUTRAL_ROOT, NEUTRAL_ROOT
    else:
        slug = _slug(title)
        root = f"{slug}-brain"

    # Resolve + validate models BEFORE any output is written.
    embed_model = str((manifest.get("embedding_model") or {}).get("name")
                      or "the embedding model named in corpus/manifest.json")
    emb_src = rr_src = None
    rs = manifest.setdefault("rag_settings", {})
    if models and cfg.semantic_search:
        emb = _resolve_shared((manifest.get("embedding_model") or {}).get("name", ""),
                              "embedder")
        if not emb:
            raise FileNotFoundError(
                "include_models requested but this corpus does not "
                "reference a shared embedder — rebuild it once from the "
                "RAG page, then export with models")
        emb_id, emb_src = emb
        # Rewrite refs for display only — server.py loads bundled models by
        # directory, never by these paths.
        manifest["embedding_model"]["name"] = emb_id
        rs["embedder"] = emb_id
        embed_model = emb_id
    rr = _resolve_or_register(rs.get("reranker", ""), "reranker") \
        if models and cfg.include_reranker and cfg.reranker_enabled else None
    if rr:
        rr_id, rr_src = rr
        rs["reranker"] = rr_id
        rs["rerank_enabled"] = True
    else:
        rs["rerank_enabled"] = False
    models = bool(emb_src or rr_src)
    manifest["search_layers"]["reranker"] = rr_src is not None   # only a bundled reranker can run

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
        chunks_df = pd.read_parquet(corpus / "chunks.parquet")
        layer_data: dict[str, bytes | Path] = {f: corpus / f for f in layer_files}
        if keep is not None:
            chunks_df, layer_data = _subset(corpus, chunks_df, layer_files, keep, manifest)
        total_docs = len({_base_id(d) for d in pd.read_parquet(corpus / "chunks.parquet",
                                                               columns=["document_id"])["document_id"]})
        manifest["scope"] = dict(scope or {"label": "all documents", "documents": total_docs, "of": total_docs})
        chunks_jsonl = chunks_df.to_json(orient="records", lines=True, force_ascii=False)
        manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
        source_files = []
        if (corpus / "sources").is_dir():
            source_files = [f for f in sorted((corpus / "sources").rglob("*"))
                            if f.is_file() and not f.is_symlink() and (keep is None or _base_id(f.stem) in keep)]
        if encrypted:
            from .brain_container import ContainerWriter
            spool = tempfile.TemporaryFile()  # noqa: SIM115 - anonymous, ciphertext only
            cw = ContainerWriter(spool, passphrase, log_n=cfg.kdf_log_n)
            cw.add("manifest.json", manifest_json.encode("utf-8"))
            cw.add("chunks.jsonl", chunks_jsonl.encode("utf-8"))
            for fname, data in layer_data.items():
                if isinstance(data, bytes):
                    cw.add(fname, data)
                else:
                    with open(data, "rb") as fp:
                        cw.add(fname, fp)
            for f in source_files:
                with open(f, "rb") as fp:
                    cw.add(f"sources/{f.relative_to(corpus / 'sources').as_posix()}", fp)
            if changes_md:
                cw.add("CHANGES.md", changes_md.encode("utf-8"))
            if ship_originals:
                cw.add(ORIGINALS_INDEX, _originals_index(ship_originals))
                for o in ship_originals:
                    with open(o.path, "rb") as fp:
                        cw.add(f"{ORIGINALS_DIR}/{o.document_id}", fp)
            cw.close()
            size = spool.tell()
            spool.seek(0)
            sink.add_stream(f"{root}/corpus/{ENC_NAME}", spool, size)
        else:
            sink.add_bytes(f"{root}/corpus/manifest.json", manifest_json)
            sink.add_bytes(f"{root}/corpus/chunks.jsonl", chunks_jsonl)
            for fname, data in layer_data.items():
                if isinstance(data, bytes):
                    sink.add_bytes(f"{root}/corpus/{fname}", data)
                else:
                    sink.add_file(f"{root}/corpus/{fname}", data)
            for f in source_files:
                sink.add_file(
                    f"{root}/corpus/sources/{f.relative_to(corpus / 'sources').as_posix()}", f)
            if ship_originals:
                sink.add_bytes(f"{root}/corpus/{ORIGINALS_INDEX}", _originals_index(ship_originals))
                for o in ship_originals:
                    sink.add_file(f"{root}/corpus/{ORIGINALS_DIR}/{o.document_id}", o.path)
        del chunks_jsonl, manifest_json, chunks_df, layer_data

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
            sink.add_bytes(f"{root}/brain_container.py",
                           _CONTAINER_SRC.read_text(encoding="utf-8"))
        sink.add_bytes(f"{root}/requirements.txt", requirements)
        # The reranker is on only when its model is inside (it said true in packages that shipped none).
        runtime = {**cfg.runtime_config(), "reranker": bool(rr_src) and cfg.reranker_enabled}
        sink.add_bytes(f"{root}/{RUNTIME_CONFIG_NAME}", json.dumps(runtime, indent=2) + "\n")
        # The same settings again, never edited: update.sh keeps only the rag.config.json keys that differ from it.
        sink.add_bytes(f"{root}/rag.config.shipped.json", json.dumps(runtime, indent=2) + "\n")
        sink.add_bytes(f"{root}/mcp-config.example.json",
                       _mcp_config_example(f"{slug}" if encrypted else f"{slug}-brain",
                                           root, encrypted))
        for sh, body in (("install.sh", _INSTALL_SH), ("run-http.sh", _RUN_HTTP_SH),
                         ("run-mcp.sh", _RUN_MCP_SH), ("setup.sh", _SETUP_SH), ("update.sh", _UPDATE_SH)):
            sink.add_bytes(f"{root}/{sh}", body, 0o755)
        if package:
            from .package_meta import public_record
            sink.add_bytes(f"{root}/package.json", json.dumps(public_record(package), indent=2) + "\n")
            sink.add_bytes(f"{root}/CHANGES.md", (changes_md_public if encrypted else changes_md) or changes_md)
        sink.add_bytes(f"{root}/README.md", _readme(
            title=title, slug=slug, embed_model=embed_model,
            encrypted=encrypted, models=models, cfg=cfg,
            bundled_embedder=emb_src is not None, bundled_reranker=rr_src is not None,
            originals=len(ship_originals), scope=manifest["scope"], package=package))
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
                         size=out.stat().st_size, passphrase=generated, layers=dict(manifest["search_layers"]),
                         runtime_search=cfg.runtime_search(bundled_embedder=emb_src is not None))

def studio_models_for(corpus_dir: str | Path) -> dict[str, Path]:
    """The studio's own copy of an index's embedder (``{"embedder": dir}``), for checking a package without models."""
    try:
        manifest = json.loads((Path(corpus_dir) / "manifest.json").read_text(encoding="utf-8"))
        found = _resolve_shared((manifest.get("embedding_model") or {}).get("name", ""), "embedder")
    except (OSError, ValueError, LookupError):
        return {}
    return {"embedder": found[1]} if found and found[1] and Path(found[1]).is_dir() else {}


def verify_package(archive: str | Path, passphrase: str | None, *, studio_models: dict[str, Path] | None = None,
                   embed_base_url: str | None = None, expected_originals: dict[str, str] | None = None,
                   out_of_scope: dict[str, str] | None = None) -> dict:
    """Open a finished package the way a customer would: unpack it, start its server, ask it something over MCP.

    Every search layer the package ships is exercised: meaning search, word search, their fusion and a bundled reranker.
    A package without its embedder inside needs one where it runs (RAG_EMBED_BASE_URL); ``studio_models`` lends the
    studio's copy of the same model for the check, so its vectors are still tested; ``embed_base_url`` points the check
    at an embedding endpoint instead. Then the package's HTTP server is
    started and its chat page and MCP-over-HTTP endpoint are checked. A package with original files has one fetched
    over MCP (``rag_get_document``) and over HTTP (``/files/<id>``), its sha256 compared with ``expected_originals``
    (``document_id -> sha256`` of the studio's file; else the package's own record). ``out_of_scope``
    (``smart_brain.scope.probe_queries``: documents left out of a scoped package, with words from their passages) are
    searched for: no hit and no listed document may come from them. Temporary files sit next to the archive and are
    removed. Returns a per-layer report.
    """
    started = time.monotonic()
    path = Path(archive)
    out = {"ok": False, "detail": "", "hits": 0, "identity": False, "layers": {}, "seconds": 0.0}
    work = Path(tempfile.mkdtemp(prefix=".verify-", dir=path.parent))
    try:
        root = _unpack_runtime(path, work)
        lent = ""
        embedder_dir = root / "corpus" / "embedder"
        studio_embedder = (studio_models or {}).get("embedder")
        if studio_embedder and not embedder_dir.exists():
            embedder_dir.symlink_to(Path(studio_embedder).resolve(), target_is_directory=True)
            lent = "embedder"
        env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_")}
        if passphrase:
            env["RAG_PASSPHRASE"] = passphrase
        if embed_base_url:
            env["RAG_EMBED_BASE_URL"] = embed_base_url
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "rag_info", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "rag_profile", "arguments": {}}},
        ]
        first = _run_mcp(root, env, requests)
        info = _tool_json(first.get(2))
        if not info:
            out["detail"] = "the server started but answered nothing usable"
            return out
        layer_status = info.get("layers", {})
        out["layers"] = {key: {"available": str(value).startswith("on"), "detail": value}
                          for key, value in layer_status.items() if key in {"semantic", "keyword", "fusion", "reranker"}}
        profile = _tool_json(first.get(3)) or {}
        query = _sample_query(root, env)
        second = _run_mcp(root, env, [requests[0], {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
            "name": "rag_search", "arguments": {"query": query, "top_k": 3}}}])
        hits = _tool_json(second.get(4))
        count = len(hits) if isinstance(hits, list) else len((hits or {}).get("hits", []))
        hits_list = hits if isinstance(hits, list) else (hits or {}).get("hits", [])
        active = {key for key in ("semantic", "keyword", "reranker")
                  if out["layers"].get(key, {}).get("available")}
        exercised = {
            "semantic": any("dense_score" in hit for hit in hits_list),
            "keyword": any("bm25_score" in hit for hit in hits_list),
            "reranker": any("ce_score" in hit for hit in hits_list),
        }
        if out["layers"].get("fusion", {}).get("available"):
            active.add("fusion")
            exercised["fusion"] = any("dense_score" in hit and "bm25_score" in hit for hit in hits_list)
        shipped_semantic = str(layer_status.get("semantic", "")) != "not in this package" \
            and not str(layer_status.get("semantic", "")).startswith("off")
        if shipped_semantic and "semantic" not in active:
            active.add("semantic")        # vectors are in the package but nothing could embed the question: a failure
        missing = sorted(layer for layer in active if not exercised.get(layer, False))
        original = None
        if int(info.get("original_files") or 0):
            original = _check_original_mcp(root, env, requests[0], expected_originals or {})
            out["original"] = original
        scope_check = _check_scope(root, env, requests[0], out_of_scope) if out_of_scope else None
        if scope_check is not None:
            out["scope"] = scope_check
        http = _check_http(root, env, original if original and original.get("ok") else None)
        out["http"] = http
        runtime_cfg = {}
        with contextlib.suppress(OSError, ValueError):
            runtime_cfg = json.loads((root / RUNTIME_CONFIG_NAME).read_text(encoding="utf-8"))
        if lent:
            out["lent"] = {"embedder": "meaning search ran with the studio's copy of the embedder: it is not inside "
                                       "this package, so where it runs it needs RAG_EMBED_BASE_URL or a rebuild with "
                                       "Include models"}
            if shipped_semantic and not runtime_cfg.get("embed_base_url") and not embed_base_url:
                out["runtime_search"] = ("where it runs: word search only, until the host sets an embeddings "
                                         "endpoint (RAG_EMBED_BASE_URL) or it is exported with Include models; "
                                         "meaning search was checked with the studio's copy of the embedder")
        out.update(hits=count, identity=bool(profile.get("instructions")),
                   exercised={layer: exercised.get(layer, False) for layer in sorted(active)},
                   ok=(count > 0 and bool(active) and not missing and http["ok"]
                       and (original is None or bool(original.get("ok")))
                       and (scope_check is None or bool(scope_check.get("ok")))),
                   detail=(f"opened it, exercised {', '.join(sorted(active))} search and found {count} passages"
                           if count and not missing else
                           f"search did not exercise available layer(s): {', '.join(missing)}"
                           if missing else "the server opened it but a search found nothing")
                   + ("" if http["ok"] else f"; HTTP server: {http['detail']}")
                   + ("" if original is None else f"; original file: {original['detail']}")
                   + ("" if scope_check is None else f"; scope: {scope_check['detail']}")
                   + (f"; {out['runtime_search']}" if out.get("runtime_search") else ""))
    except (OSError, subprocess.SubprocessError, ValueError, tarfile.TarError, zipfile.BadZipFile) as e:
        out["detail"] = f"could not open the package: {type(e).__name__}: {str(e)[:200]}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
        out["seconds"] = round(time.monotonic() - started, 1)
    return out


def _unpack_runtime(archive: Path, dest: Path) -> Path:
    """Unpack the complete package runtime, including optional bundled model folders."""
    def wanted(name: str) -> bool:
        parts = Path(name).parts
        return ".." not in parts and not name.startswith("/")

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest, [n for n in zf.namelist() if wanted(n)])
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(dest, members=[m for m in tf.getmembers() if wanted(m.name) and (m.isfile() or m.isdir())],
                          filter="data")
    server = next(dest.rglob("server.py"), None)
    if server is None:
        raise ValueError("no server.py in the package")
    return server.parent


def _run_mcp(root: Path, env: dict, requests: list[dict]) -> dict:
    """Send JSON-RPC lines to the package's real server; {id: response}."""
    args = [sys.executable, "-I", str(root / "server.py"), "--mcp"]
    done = subprocess.run(args, input="".join(json.dumps(r) + "\n" for r in requests), capture_output=True, text=True,
                          env=env, cwd=root, timeout=120, check=False)
    if done.returncode != 0:
        raise ValueError((done.stderr or "the server exited with an error").strip().splitlines()[-1])
    replies = [json.loads(line) for line in done.stdout.splitlines() if line.strip().startswith("{")]
    return {r.get("id"): r for r in replies}


def _check_scope(root: Path, env: dict, init: dict, probes: dict[str, str]) -> dict:
    """Search for words of documents left out of the package: none of them may come back, nor be listed."""
    calls = [init, {"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {
        "name": "rag_list_documents", "arguments": {}}}]
    calls += [{"jsonrpc": "2.0", "id": 11 + i, "method": "tools/call", "params": {
        "name": "rag_search", "arguments": {"query": q, "top_k": 20}}} for i, q in enumerate(probes.values())]
    replies = _run_mcp(root, env, calls)
    outside = set(probes)
    listed = {_base_id(d.get("document_id") or "") for d in (_tool_json(replies.get(10)) or [])}
    leaked = listed & outside
    for i in range(len(probes)):
        hits = _tool_json(replies.get(11 + i)) or []
        leaked |= {_base_id(h.get("document_id") or "") for h in hits} & outside
    if leaked:
        return {"ok": False, "detail": f"{len(leaked)} document(s) outside the scope can be found"}
    return {"ok": True, "detail": f"{len(probes)} search(es) for left-out documents found nothing outside the scope"}


def _check_original_mcp(root: Path, env: dict, init: dict, expected: dict[str, str]) -> dict:
    """Fetch the smallest original file over MCP and compare its sha256 with the studio's file."""
    import base64
    import hashlib

    listed = _tool_json(_run_mcp(root, env, [init, {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {
        "name": "rag_list_documents", "arguments": {}}}]).get(5)) or []
    shipped = sorted((d for d in listed if d.get("original")), key=lambda d: int(d.get("size") or 0))
    if not shipped:
        return {"ok": False, "detail": "rag_list_documents lists no original file"}
    doc = shipped[0]
    reply = _run_mcp(root, env, [init, {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {
        "name": "rag_get_document", "arguments": {"document_id": doc["document_id"]}}}]).get(6) or {}
    content = (reply.get("result") or {}).get("content") or []
    try:
        meta = json.loads(content[0]["text"])
        blob = next(c["resource"]["blob"] for c in content if c.get("type") == "resource")
    except (IndexError, KeyError, ValueError, StopIteration, TypeError):
        return {"ok": False, "detail": f"rag_get_document returned no file for {doc.get('filename')!r}"}
    got = hashlib.sha256(base64.b64decode(blob)).hexdigest()
    want = expected.get(doc["document_id"]) or meta.get("sha256")
    ok = got == want
    return {"ok": ok, "document_id": doc["document_id"], "filename": doc.get("filename"), "sha256": got,
            "detail": (f"{doc.get('filename')} came back over MCP unchanged (sha256 matches)" if ok else
                       f"{doc.get('filename')} over MCP differs from the source (sha256 {got[:12]} != {str(want)[:12]})")}


def _check_http(root: Path, env: dict, original: dict | None = None) -> dict:
    """Start the package's HTTP server on a free loopback port: the chat page loads and MCP over HTTP answers."""
    import socket
    import urllib.error
    import urllib.request

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = root / ".http-check.log"   # a file, not a pipe: the server logs every request and a full pipe blocks it
    with open(log, "w") as log_fp:
        proc = subprocess.Popen([sys.executable, "-I", str(root / "server.py"), "--http", str(port), "--host", "127.0.0.1"],
                                cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log_fp)
    base = f"http://127.0.0.1:{port}"
    try:
        page = ""
        for _ in range(240):            # an encrypted corpus is decrypted at start
            if proc.poll() is not None:
                err = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                return {"ok": False, "detail": "it exited: " + (err[-1] if err else f"code {proc.returncode}")}
            try:
                with urllib.request.urlopen(base + "/", timeout=2) as r:
                    page = r.read().decode("utf-8", "replace")
                break
            except urllib.error.HTTPError as e:   # it answers, but wrongly: waiting longer will not help
                return {"ok": False, "detail": f"the chat page answered HTTP {e.code}"}
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        if 'src="/app.js"' not in page:
            return {"ok": False, "detail": "the chat page did not load"}
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "server/discover", "params": {"_meta": {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28"}}}).encode()
        req = urllib.request.Request(base + "/mcp", data=body, method="POST", headers={
            "Content-Type": "application/json", "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "server/discover"})
        with urllib.request.urlopen(req, timeout=10) as r:
            found = json.loads(r.read())
        if "2026-07-28" not in (found.get("result") or {}).get("supportedVersions", []):
            return {"ok": False, "detail": "MCP over HTTP did not answer server/discover"}
        if original:
            import hashlib
            with urllib.request.urlopen(f"{base}/files/{original['document_id']}", timeout=30) as r:
                got = hashlib.sha256(r.read()).hexdigest()
                attached = "attachment" in (r.headers.get("Content-Disposition") or "")
            if got != original["sha256"] or not attached:
                return {"ok": False, "detail": f"GET /files/{original['document_id']} did not return the original file"}
            return {"ok": True, "detail": "chat page, MCP over HTTP and the original-file download answer"}
        return {"ok": True, "detail": "chat page and MCP over HTTP answer"}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:160]}"}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _tool_json(reply: dict | None):
    try:
        return json.loads(reply["result"]["content"][0]["text"])  # type: ignore[index]
    except (TypeError, KeyError, IndexError, ValueError):
        return None


def _sample_query(root: Path, env: dict) -> str:
    """A few words from the package's own first passage, read through the server so encrypted packages work too."""
    code = ("import server,json,os;c=server.Corpus(server.CORPUS_DIR,server.resolve_config({},"
            "os.environ,{})[0],os.environ.get('RAG_PASSPHRASE'));print(' '.join(c.chunks[0]['text'].split()[:12]))")
    done = subprocess.run([sys.executable, "-I", "-c", f"import sys;sys.path.insert(0,'.');{code}"], capture_output=True,
                          text=True, env=env, cwd=root, timeout=120, check=False)
    return done.stdout.strip() or "the"
