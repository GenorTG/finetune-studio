# Restart — Finetune Studio

## Install

```bash
# Linux/macOS (GPU auto-detected, GGUF included by default):
cd finetune-studio
bash install.sh

# Verify:
bash install.sh --check

# CPU-only fallback (no GPU):
bash install.sh --cpu

# Conda:
USE_CONDA=1 bash install.sh

# Fish:
fish install.fish
```

## Start WebUI

```bash
# Quick (auto-installs if venv missing):
bash run.sh

# Manual:
source .venv/bin/activate
uvicorn finetune_studio.webui.app:app --host 0.0.0.0 --port 7860
```

## Install flags

| Flag | Effect |
|------|--------|
| (none) | GPU-aware: detects NVIDIA/AMD/Intel, installs CUDA wheels, GGUF included |
| `--cpu` | Force CPU-only (skip GPU detection) |
| `--no-gguf` | Skip llama-cpp-python |
| `--check` | Verify install (no changes) |

## GPU wheel selection

| GPU | PyTorch | llama-cpp-python |
|-----|---------|-----------------|
| NVIDIA driver 550+ | cu130 from pytorch.org | cu130 from abetlen CUDA index |
| NVIDIA driver 525+ | cu124 | cu124 |
| NVIDIA driver 470+ | cu118 | cu118 |
| AMD ROCm | rocm6.2 | CPU (ROCm llama wheels not available) |
| Intel XPU | xpu | CPU |
| No GPU | CPU | CPU |

## Makefile targets

```bash
make install       # GPU-aware install
make install-cpu   # CPU-only
make install-conda # Conda
make check         # Verify
make run           # Start webui
make test          # pytest
make lint          # ruff check
```

## Fan-dragon

The WebUI runs as a **systemd user unit** `finetune-studio` (no sudo; linger is on, so it survives logout/reboot). Logs go to the user journal.

### First-time install

```bash
ssh fan-dragon 'bash -lc "cd ~/finetune-studio && bash install.sh && bash install-service.sh"'
```

`install-service.sh` writes `~/.config/systemd/user/finetune-studio.service`, removes the legacy `finetune-studio-webui.service`, stops any bare uvicorn holding :7860, starts the unit and polls `/api/providers`. Re-run it after changing the unit template, and once on every host that still runs the pre-2026-10-10 unit (a bare uvicorn) so systemd runs the supervisor instead (`update.sh` warns while it does not).

### Deploy a new commit

```bash
ssh fan-dragon 'bash -lc "cd ~/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio"'
```

### Verify

```bash
ssh fan-dragon 'bash -lc "systemctl --user is-active finetune-studio; P=\$(ss -ltnpH sport = :7860 | grep -oP pid=\\K[0-9]+); echo pid \$P; grep finetune-studio.service /proc/\$P/cgroup"'
ssh fan-dragon 'journalctl --user -u finetune-studio -n 50 --no-pager'
```

Expect `active`, and the :7860 pid's cgroup line containing `finetune-studio.service`. If the pid is in a `session-*.scope` instead, a bare uvicorn is squatting the port — re-run `bash install-service.sh`.

## Windows

```powershell
.\install.ps1
.\run.bat
```

## Supervisor (from 2026-10-10)

The supervisor owns :7860 and runs the web UI as its child, restarting it with backoff when it dies or stops answering `/api/health`. Start it either way (both are first-class):

```bash
fts up --manual       # detached supervisor, no systemd; log in $FTS_ROOT/run/supervisor.log  (or `fts supervisor` in a terminal)
fts service install   # once: write + enable the systemd user unit (wraps install-service.sh; --no-start to only install)
fts up --systemd      # start the unit.   `fts up` alone picks systemd when installed, else manual
fts down              # stop it, whichever way it was started
fts service uninstall | status | restart
```

Operate it with `fts` (no HTTP needed) or from Settings -> Service in the web UI:

```bash
fts status            # component table; exit 0 all ready, 1 degraded, 3 supervisor down
fts restart web       # only the web child: ~6 s, port stays open, applies a changed Settings -> Compute device
fts logs web -n 100   # child output tail;  fts events -f  # spawns, exits (with signal), health, backoff
fts doctor            # works with everything down: supervisor, port, unit, compute-device file, last event
```

`systemctl --user restart finetune-studio` restarts the whole tree (needed when supervisor code itself changed; `fts restart web` only reloads the web child). A manual supervisor started inside another unit's cgroup dies with that unit. Plain `run.sh` / `fts webui` run without a supervisor (`/api/service/status` -> `managed: false`).
