# Installation Guide

Install Finetune Studio on **Linux** (Ubuntu / Debian, Fedora, Arch), with an NVIDIA GPU or on the CPU. **Linux is the only supported platform** until the stable 1.0.0 release; Windows and macOS are not supported or tested.

Releases are tagged `vX.Y.Z` on the `main` branch ([CHANGELOG](../CHANGELOG.md), [GitHub Releases](https://github.com/GenorTG/finetune-studio/releases)); `main` always holds the latest release.

---

## Quick start — Linux + NVIDIA GPU

```bash
git clone https://github.com/GenorTG/finetune-studio.git
cd finetune-studio
./install.sh                       # does everything below
source .venv/bin/activate
fts webui                          # address: Settings -> Access & network (default 127.0.0.1:7860)
```

Then open http://localhost:7860.

If something fails, read the relevant section below.

---

## Prerequisites — what you need first

| Prerequisite | Why | Install |
|---|---|---|
| **Python 3.12 or 3.13** | Supported runtime range (`>=3.12,<3.14`) | https://www.python.org/downloads/ |
| **Git** | Clones the repo, pulls updates | https://git-scm.com/downloads |
| **NVIDIA GPU + current driver** (recommended) | Fast training & inference; the installer selects a compatible PyTorch wheel | https://www.nvidia.com/drivers |
| **CUDA Toolkit** (optional) | Needed only when a GPU-enabled llama.cpp component must be built from source; not required for PyTorch prebuilt CUDA wheels | https://developer.nvidia.com/cuda-toolkit |
| **Node.js ≥ 18** (optional) | Only needed if you want to run the Playwright UI tests yourself | https://nodejs.org/ |
| **15 GB free disk** | PyTorch + transformers + model weights | — |

### How to install each prerequisite from the official source

**Python**
- `sudo apt install python3 python3-venv python3-dev` (Debian/Ubuntu), `sudo dnf install python3 python3-devel` (Fedora) or `sudo pacman -S python` (Arch)

**Git**
- `apt install git` / `dnf install git` / `pacman -S git` (or https://git-scm.com/downloads)

**NVIDIA driver**
- Install a current driver from https://www.nvidia.com/drivers.
- Verify with `nvidia-smi` — it should print your GPU name, driver version, and supported CUDA runtime level.
- The CUDA Toolkit is not needed for PyTorch's prebuilt wheels. Install it only if you need to compile a CUDA-enabled llama.cpp component from source.
- If you have an AMD GPU, see the **AMD / CPU-only** section below.

**Node.js** (only for running tests)
- download LTS from https://nodejs.org/ — or `apt install nodejs npm`

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
1. Detect the GPU (NVIDIA; AMD / Intel are best effort; CPU only when there is none) and create a Python venv in `.venv/` (with `uv`)
2. Install the newest PyTorch build your driver supports and pin it (`.venv/torch-constraints.txt`)
3. Install `llama-cpp-python` with GPU offload, the app dependencies and the `parsers` extra used for document ingestion
4. Build the llama.cpp CLI (`.llama.cpp/`, used for GGUF export) for the same GPU backend
5. Run the install-time health check and print next steps

Useful flags (`./install.sh --help` lists all):

| Flag | What it does |
|---|---|
| `--plan` | print the detected hardware and the install plan, change nothing |
| `--check` / `--verify` | health check, no changes (`--verify` also checks the systemd unit; exit code = health) |
| `--repair` | diagnose and fix a broken install (mixed torch, missing deps, CPU build on a GPU host) |
| `--cpu`, `--gpu <vendor>` | override detection. The choice is remembered in `.venv/fts-force-vendor`, so later `--check`, `--repair` and `update.sh` keep it; run `--gpu <vendor>` to change it |

On Linux, `./update.sh` pulls, checks/repairs the environment, syncs dependencies, migrates the database, and restarts the service running this checkout; `./update.sh --check` is the read-only mode. Install the optional systemd user service with `./install-service.sh`.

The installer does not install Playwright or browser binaries. See
[`../tests/README_E2E.md`](../tests/README_E2E.md) for the separate browser-test
setup.

If `./install.sh` doesn't exist or is broken, do it manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
# GPU install: do not hard-code a CUDA wheel index; it depends on the driver and GPU.
# Run ./install.sh --plan and use its torch_index, or let the installer choose it.
# Example shape only: python -m pip install torch --index-url <torch_index from the plan>
# CPU-only:
# python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[parsers]"
```

### 3. Install llama-cpp-python (only if you want GGUF inference)

The base install gives you `transformers`-based inference. To run GGUF
quantized models (fast, low-VRAM), you need `llama-cpp-python`. GPU
build requires CUDA:

```bash
# CPU-only GGUF:
python -m pip install llama-cpp-python

# NVIDIA GPU GGUF (requires the CUDA toolkit):
CMAKE_ARGS="-DGGML_CUDA=on" python -m pip install llama-cpp-python --force-reinstall --no-cache-dir

```

### 4. Start the studio

```bash
./run.sh                          # or: source .venv/bin/activate && fts webui
bash install-service.sh           # optional systemd user service
```

Open http://localhost:7860 in a modern browser. The first visit offers the onboarding tutorial.

### 5. Access: who can open the web UI

By default Finetune Studio is reachable from **this computer only (127.0.0.1:7860), no password**. Who else can
open it is a set of switches; anything beyond this computer needs a password:

| Switch | Who | Listens on |
|---|---|---|
| This computer | always on | 127.0.0.1 |
| Local network | devices on your LAN / Wi-Fi | 0.0.0.0, and refuses what is switched off |
| Tailscale | your [Tailscale](https://tailscale.com) devices (offered when Tailscale is detected) | the Tailscale address only |
| Custom addresses | any other address of this computer (a VPN, `::`, `0.0.0.0` = every network) | those addresses |

A terminal install asks one question per switch (Tailscale only when it is detected), then a password if any is on.
Change them any time in **Settings -> Access & network**: switching off works at once; switching on, or a new port,
applies with **Apply now** (restarts the web UI in a few seconds; training keeps running). On the machine itself:

```bash
fts server show                   # each switch, this computer's LAN / Tailscale addresses, what is bound
fts server set --lan on           # devices on the local network (asks for a password first)
fts server set --tailscale on     # your Tailscale devices
fts server set --custom 198.51.100.2   # other addresses ('' clears); --port N; --allowed-hosts for a reverse proxy
fts restart web                   # apply switching on / a new port
fts auth status                   # password on/off, who can reach it, sessions, API tokens
fts auth set-password             # turn sign-in on / change the password (prompt; or --password-file FILE)
fts auth token create "backup"    # API token for scripts: Authorization: Bearer <token> (shown once)
fts auth disable                  # no sign-in (only sensible for this computer only)
```

Forgot the password? Run `fts auth set-password` on the machine. A web UI reachable from other machines
without a password shows a red warning on every page, and `fts doctor` reports it as a failure. If a new port is
taken, the web UI keeps the previous one and Settings says why.

**Automated installs** pass a config file instead of answering (`bash install.sh --config FILE`, or
`FTS_INSTALL_CONFIG=FILE`). The file is parsed, never executed; an unknown key is an error:

```ini
FTS_LISTEN_LAN=1                           # devices on the local network / Wi-Fi
FTS_TAILSCALE=1                            # your Tailscale devices
FTS_LISTEN_CUSTOM=198.51.100.2             # other addresses, comma separated (0.0.0.0 = every network)
FTS_PORT=7860
FTS_AUTH=password                          # password | off (off is refused with any switch on)
FTS_ADMIN_PASSWORD_FILE=/run/secrets/fts   # or FTS_ADMIN_PASSWORD=... in this file (chmod 600 it)
FTS_ALLOWED_HOSTS=studio.example           # extra host names for a reverse proxy, comma separated
FTS_INSTALL_SERVICE=1                      # also install the systemd user unit
```

`FTS_BIND_HOST` (the older single address; `0.0.0.0` = local network + Tailscale) still works.

State lives under `$FTS_ROOT` (default `~/.finetune-studio`): `server.json` (the switches and port) and `auth/` (password hash,
API token hashes, sessions; mode 0600). No plain password or token is ever stored.

---

## Per-distribution notes

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

---

## AMD GPUs (Linux)

Finetune Studio's training path uses PyTorch ROCm. To install:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
python -m pip install -e ".[parsers]"
```

Inference via GGUF (llama-cpp-python) requires a custom build with HIP
support — see https://github.com/abetlen/llama-cpp-python#hipblas

---

## CPU-only (no GPU)

Works fine for small models and inference. Training is slow.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[parsers]"
python -m pip install llama-cpp-python    # GGUF CPU inference
```

---

## Verifying the install

Run the environment check from the checkout:

```bash
source .venv/bin/activate
fts doctor
```

Then start the WebUI (`fts webui`) and open http://localhost:7860. The default listener is local-only; enable LAN/Tailscale access in **Settings → Access & network** only when needed.

---

## Common install issues

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR: No matching distribution found for torch` | Wrong Python version | Use Python 3.12 or 3.13 |
| `nvcc: command not found` when building llama-cpp | CUDA toolkit not installed | Install CUDA toolkit 12.x |
| Training is invisible / no progress | GPU not detected by torch | Reinstall torch with correct CUDA index URL |
| `libstdc++.so.6: version not found` | Old base image | `conda install -c conda-forge libstdcxx-ng` or use newer Ubuntu |
| HF downloads fail | Firewall or proxy | Set `HF_HUB_OFFLINE=1` and pre-download, or configure proxy |
| WebUI loads but the Smart Brain build OOMs | Not enough RAM/VRAM | Use a smaller embedder (e.g. `all-MiniLM-L6-v2`) |

---

## Updating

```bash
cd finetune-studio
bash update.sh          # pulls main (the latest release), repairs the venv, syncs deps, migrates, restarts
bash update.sh --check  # read-only: what would change
```

Settings → **Apply update** runs the same script from the web UI. To pin one release instead:
`git fetch --tags && git checkout vX.Y.Z && bash update.sh --no-pull`. What changed: [CHANGELOG](../CHANGELOG.md).

Models and Smart Brain indexes live in `~/.finetune-studio/` — they persist across
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
| Smart Brain indexes | `~/.finetune-studio/brain/` |
| VRAM profile dumps | `~/.cache/fts-vram-profile/` |

See `docs/ATTRIBUTIONS.md` for the full list of open-source packages used.
