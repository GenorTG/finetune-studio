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

systemd runs `python -m finetune_studio.supervisor`; it owns :7860 and runs the web UI as its child, restarting it with backoff when it dies or stops answering `/api/health`. Operate it with `fts` (needs no HTTP):

```bash
fts status            # component table; exit 0 all ready, 1 degraded, 3 supervisor down
fts restart web       # restart only the web child: ~3 s, port stays open, picks up a changed Settings -> Compute device
fts logs web -n 100   # child output tail;  fts events -f  # spawns, exits (with signal), health, backoff
fts doctor            # works with everything down: supervisor, port, unit, compute-device file, last event
fts up                # supervisor down: start the systemd unit
```

`systemctl --user restart finetune-studio` still restarts the whole tree. Plain `run.sh` / `fts webui` run without a supervisor (`GET /api/system/supervisor` -> `managed: false`).
