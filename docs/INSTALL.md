# Installation Guide

Step-by-step instructions for getting Finetune Studio running on **Linux**
(Ubuntu / Fedora / Arch), **Windows 10/11**, and **macOS**.

---

## TL;DR — the 30-second version (Linux + NVIDIA GPU)

```bash
git clone https://github.com/GenorTG/finetune-studio.git
cd finetune-studio
./install.sh                       # does everything below
source .venv/bin/activate
fts web --host 0.0.0.0 --port 7860
```

Then open http://localhost:7860.

If something fails, read the relevant section below.

---

## Prerequisites — what you need first

| Prerequisite | Why | Install |
|---|---|---|
| **Python ≥ 3.10** | Required by transformers/peft/torch | https://www.python.org/downloads/ |
| **Git** | Clones the repo, pulls updates | https://git-scm.com/downloads |
| **NVIDIA GPU + CUDA driver** (recommended) | Fast training & inference | https://www.nvidia.com/drivers |
| **NVIDIA CUDA Toolkit 12.x** | Required to build PyTorch + llama-cpp-python with GPU support | https://developer.nvidia.com/cuda-toolkit |
| **Node.js ≥ 18** (optional) | Only needed if you want to run the Playwright UI tests yourself | https://nodejs.org/ |
| **15 GB free disk** | PyTorch + transformers + model weights | — |

### How to install each prerequisite from the official source

**Python**
- Linux:   `sudo apt install python3 python3-venv python3-dev` (Debian/Ubuntu) or `sudo dnf install python3 python3-devel` (Fedora) or `sudo pacman -S python` (Arch)
- macOS:   `brew install python@3.12`  (https://brew.sh) — or download from https://www.python.org/downloads/macos/
- Windows: download the official installer from https://www.python.org/downloads/windows/ — **make sure to check "Add python.exe to PATH"**

**Git**
- All OSes: download from https://git-scm.com/downloads — or `apt install git` / `brew install git`

**NVIDIA driver + CUDA**
- NVIDIA driver: https://www.nvidia.com/drivers — pick your GPU + OS, download the .run / .exe / .dmg
- CUDA Toolkit 12.x: https://developer.nvidia.com/cuda-12-x-downloads (Linux / Windows / macOS tabs)
- After install, verify with `nvidia-smi` — should print your GPU name + driver version + CUDA version
- If you have an AMD GPU, see the **AMD / CPU-only** section below.

**Node.js** (only for running tests)
- All OSes: download LTS from https://nodejs.org/ — or `brew install node` / `apt install nodejs npm`

---

## Step-by-step install

### 1. Clone

```bash
git clone https://github.com/GenorTG/finetune-studio.git
cd finetune-studio
```

### 2. Run the installer

```bash
chmod +x install.sh
./install.sh
```

This will:
1. Create a Python venv in `.venv/`
2. Upgrade pip
3. Install PyTorch (GPU or CPU variant based on your hardware)
4. Install all dependencies from `pyproject.toml`
5. Install Playwright browsers (only if Node is detected)
6. Print a "next steps" message

If `./install.sh` doesn't exist or is broken, do it manually:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip wheel
# GPU (NVIDIA) install — pick the CUDA version that matches your driver:
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU-only:
# python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

### 3. Install llama-cpp-python (only if you want GGUF inference)

The base install gives you `transformers`-based inference. To run GGUF
quantized models (fast, low-VRAM), you need `llama-cpp-python`. GPU
build requires CUDA:

```bash
# CPU-only GGUF:
python -m pip install llama-cpp-python

# NVIDIA GPU GGUF (Linux/Windows, requires CUDA toolkit installed):
CMAKE_ARGS="-DGGML_CUDA=on" python -m pip install llama-cpp-python --force-reinstall --no-cache-dir

# Apple Silicon (Metal):
CMAKE_ARGS="-DGGML_METAL=on" python -m pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 4. Start the studio

```bash
source .venv/bin/activate
fts web --host 0.0.0.0 --port 7860
# or directly:
python -m finetune_studio.webui.app
```

Open http://localhost:7860 in any modern browser. First visit triggers the
onboarding tutorial.

---

## Per-OS notes

### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev build-essential git curl
# Install NVIDIA driver (Ubuntu 22.04+ usually has the "Additional Drivers" tab
# in Software & Updates). For headless servers:
sudo apt install -y nvidia-driver-550          # check nvidia.com for current
sudo reboot                                    # driver activation
nvidia-smi                                     # verify
# Install CUDA toolkit (optional, only if building llama-cpp-python with GPU):
sudo apt install -y cuda-toolkit-12-3
```

### Linux (Fedora / RHEL)

```bash
sudo dnf install -y python3 python3-devel gcc gcc-c++ git curl
sudo dnf install -y akmod-nvidia              # NVIDIA driver
sudo dnf install -y cuda-toolkit
```

### Linux (Arch / Manjaro)

```bash
sudo pacman -S python python-pip git base-devel
sudo pacman -S nvidia nvidia-utils cuda
```

### macOS (Intel / Apple Silicon)

```bash
# Install Homebrew first: https://brew.sh
brew install python@3.12 git cmake

# For Apple Silicon (M1/M2/M3/M4) — Metal acceleration works automatically.
# For Intel Macs — inference is CPU-only.

# No NVIDIA driver needed. CUDA toolkit not needed.
```

### Windows 10 / 11

1. Install Python 3.12 from python.org (check "Add to PATH")
2. Install Git from git-scm.com
3. Install the NVIDIA driver + CUDA 12.x from nvidia.com
4. Open PowerShell:
   ```powershell
   git clone https://github.com/GenorTG/finetune-studio.git
   cd finetune-studio
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
   python -m pip install -e .
   fts web --host 0.0.0.0 --port 7860
   ```
5. Allow Python through Windows Firewall when prompted
6. Open http://localhost:7860

---

## AMD GPUs (Linux)

Finetune Studio's training path uses PyTorch ROCm. To install:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
python -m pip install -e .
```

Inference via GGUF (llama-cpp-python) requires a custom build with HIP
support — see https://github.com/abetlen/llama-cpp-python#hipblas

---

## CPU-only (no GPU)

Works fine for small models and inference. Training is slow.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m pip install llama-cpp-python    # GGUF CPU inference
```

---

## Verifying the install

After install, run this 30-second smoke test:

```bash
source .venv/bin/activate
python -c "
import torch, transformers, peft, fastapi
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('fastapi:', fastapi.__version__)
print('CUDA available:', torch.cuda.is_available())
"
fts web --host 0.0.0.0 --port 7860 &
sleep 5
curl -s http://localhost:7860/api/projects | head -c 200
```

If you see a JSON array (or `[]`), you're good. Open http://localhost:7860.

---

## Common install issues

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR: No matching distribution found for torch` | Wrong Python version | Use Python 3.10–3.13 |
| `nvcc: command not found` when building llama-cpp | CUDA toolkit not installed | Install CUDA toolkit 12.x |
| Training is invisible / no progress | GPU not detected by torch | Reinstall torch with correct CUDA index URL |
| `libstdc++.so.6: version not found` | Old base image | `conda install -c conda-forge libstdcxx-ng` or use newer Ubuntu |
| HF downloads fail | Firewall or proxy | Set `HF_HUB_OFFLINE=1` and pre-download, or configure proxy |
| WebUI loads but RAG build OOMs | Not enough RAM/VRAM | Use a smaller embedder (e.g. `all-MiniLM-L6-v2`) |
| Windows: `failed to build llama-cpp-python` | Missing Visual Studio Build Tools | Install "Desktop development with C++" from Visual Studio Installer |
| macOS: `Library not loaded: libomp.dylib` | Missing OpenMP runtime | `brew install libomp` |

---

## Updating

```bash
cd finetune-studio
git pull
source .venv/bin/activate
python -m pip install -e . --upgrade
```

Models and RAG corpora live in `~/.finetune-studio/` — they persist across
updates.

---

## Uninstalling

```bash
# Remove the app directory (your data lives in ~/.finetune-studio):
rm -rf finetune-studio

# Optionally remove all data:
rm -rf ~/.finetune-studio
rm -rf ~/.cache/huggingface      # downloaded models
```

---

## Where things live after install

| What | Where |
|---|---|
| Project data | `~/.finetune-studio/projects.db` |
| Shared model weights | `~/.finetune-studio/shared_models/` |
| HuggingFace cache | `~/.cache/huggingface/` |
| Embedded vector stores | `~/.finetune-studio/rag/` |
| VRAM profile dumps | `~/.cache/fts-vram-profile/` |

See `docs/ATTRIBUTIONS.md` for the full list of open-source packages used.
