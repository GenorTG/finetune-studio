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

```bash
ssh genortg@100.125.137.96
cd ~/finetune-studio
bash install.sh  # first time only
bash run.sh      # start webui on :7860
```

## Windows

```powershell
.\install.ps1
.\run.bat
```
