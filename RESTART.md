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

**Reality check first:** fan-dragon has NO `finetune-studio.service`. The WebUI runs as a bare `uvicorn` process. Before restarting, check what's actually serving 7860:

```bash
ssh fan-dragon 'ss -ltnp | grep 7860'
```

If something is already listening (e.g. `users:(("python",pid=939407,fd=16))`), that's the live process. Restart it; don't start a second one.

### First-time install

```bash
ssh fan-dragon 'cd ~/finetune-studio && bash install.sh'
```

### Start / restart the WebUI on :7860

The pkill-then-start ordering is unreliable on fan-dragon (the `pkill -f "uvicorn finetune_studio"` regex matches the just-spawned child too, races the SSH session, and exits 255). Use start-then-kill-old instead, identifying the existing pid first:

```bash
# Step 1 — capture the currently-serving pid (if any)
OLD=$(ssh fan-dragon 'ss -ltnp | grep 7860 | grep -oP "pid=\K[0-9]+" | head -1')
echo "current pid: $OLD"

# Step 2 — start a new uvicorn, wait for it to bind, then kill only the old pid
ssh fan-dragon 'bash -lc "\
  cd /home/genortg/finetune-studio && \
  nohup .venv/bin/python -m uvicorn finetune_studio.webui.app:app \
    --host 0.0.0.0 --port 7860 --no-access-log \
    </dev/null >/tmp/uvicorn.log 2>&1 & \
  sleep 3 && \
  kill '"$OLD"' 2>/dev/null ; \
  sleep 1 && \
  ss -ltnp | grep 7860 && \
  tail -4 /tmp/uvicorn.log"'
```

The `</dev/null` keeps uvicorn alive after the SSH session closes; `nohup` detaches it; `--no-access-log` cuts noise (you have the JSON log at `/tmp/uvicorn.log`); `bash -lc` is required because fish on fan-dragon doesn't expand `$!`.

Expect `ss` to show a single `python` PID listening, and the new pid must differ from `$OLD`. If the new pid equals the old pid, the kill step never ran.

### Verify

```bash
ssh fan-dragon 'tail -10 /tmp/uvicorn.log; ss -ltnp | grep 7860'
```

Expect last log line `Uvicorn running on http://0.0.0.0:7860` and a single `python` PID listening. New pid ≠ old pid = restart actually worked.

### Pull a new commit before restart

```bash
ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'
```

Then restart per the command above.

### Install systemd unit (optional)

If you want auto-restart on crash / boot, install the unit:

```bash
scp scripts/finetune-studio.service fan-dragon:~/finetune-studio/scripts/
ssh fan-dragon 'sudo cp ~/finetune-studio/scripts/finetune-studio.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now finetune-studio'
```

Until then, the bare uvicorn above is the documented way.

## Windows

```powershell
.\install.ps1
.\run.bat
```
