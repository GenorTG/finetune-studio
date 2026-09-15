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

`install-service.sh` writes `~/.config/systemd/user/finetune-studio.service`, removes the legacy `finetune-studio-webui.service`, stops any bare uvicorn holding :7860, starts the unit and polls `/api/providers`. Re-run it after changing the unit template.

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
