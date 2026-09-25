#!/usr/bin/env bash
# Finetune Studio — GPU-aware installer.
#
# Default:
#   1. Detects GPU vendor (NVIDIA / AMD ROCm / Intel XPU / CPU)
#   2. Python 3.12+ required
#   3. Creates .venv + installs PyTorch with matching GPU wheels
#   4. Installs llama-cpp-python with CUDA/ROCm prebuilt wheels (GGUF default)
#   5. Installs everything else from pyproject.toml
#
# Flags:
#   --check         verify install, no changes
#   --repair        diagnose broken install + auto-fix (mixed torch, missing deps, etc.)
#   --verify        deep diagnostic, exit code = health (2=critical, 1=warn, 0=ok)
#   --auto-repair   like --repair but runs inline after a fresh install too
#                   (so the post-install health check auto-fixes any GPU/torch/
#                   llama-cpp/CUDA-toolkit/llama.cpp-CLI issue it finds instead
#                   of just warning). Equivalent to FTS_AUTO_REPAIR=1.
#   --cpu           force CPU-only (skip GPU wheel selection)
#   --no-gguf       skip llama-cpp-python entirely
#   --help      show usage

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PYTHON_VERSION="${PYTHON_VERSION:-}"
USE_CONDA="${USE_CONDA:-0}"
CONDA_ENV="${CONDA_ENV:-chris-ai}"
VENV_DIR="${VENV_DIR:-.venv}"
# llama.cpp lives inside the project, NOT in $HOME. Boxed-in so the install
# is reproducible and self-contained — nothing references external paths.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PROJECT_ROOT/.llama.cpp}"
# If the user still has a legacy install at $HOME/llama.cpp, warn once
# so they know to migrate manually (we never auto-move external dirs).
if [ -d "$HOME/llama.cpp" ] && [ "$HOME/llama.cpp" != "$LLAMA_CPP_DIR" ]; then
    warn "Found legacy llama.cpp checkout at $HOME/llama.cpp."
    warn "New installs use the project-local path: $LLAMA_CPP_DIR"
    warn "To migrate:  mv $HOME/llama.cpp $LLAMA_CPP_DIR"
fi
FORCE_CPU=0
SKIP_GGUF=0
SKIP_LLAMA_CPP=0
LLAMA_CPP_ONLY=0

log()  { echo "[install] $*"; }
warn() { echo "[install] WARN: $*" >&2; }
die()  { echo "[install] ERROR: $*" >&2; exit 1; }

for arg in "$@"; do
    case "$arg" in
        --check)          MODE="check" ;;
        --repair)         MODE="repair" ;;
        --verify)         MODE="verify" ;;
        --auto-repair)    AUTO_REPAIR=1 ;;
        --cpu)            FORCE_CPU=1 ;;
        --no-gguf)        SKIP_GGUF=1 ;;
        --no-llama-cpp)   SKIP_LLAMA_CPP=1 ;;
        --llama-cpp-only) LLAMA_CPP_ONLY=1 ;;
        --help|-h) sed -n '2,18p' "$0" | sed 's/^# *//'; exit 0 ;;
        *) die "Unknown arg: $arg  (try --help)" ;;
    esac
done
: "${MODE:=install}"

# ── Delegate to Python diagnostic for these modes ──────────────────────
# The Python module (scripts/install_diagnose.py) owns the deep health
# logic. Bash handles package installs + cmake builds; Python inspects.
DIAGNOSE_PY="$(dirname "$(readlink -f "$0")")/scripts/install_diagnose.py"
run_diagnose() {
    local extra_args=("$@")
    if [ -x "$VENV_DIR/bin/python" ] && [ "$MODE" != "repair" ]; then
        # Use the venv python if it's at least importable; else fall back
        # to the system python (the helper avoids importing torch itself).
        if "$VENV_DIR/bin/python" -c "import sys" >/dev/null 2>&1; then
            "$VENV_DIR/bin/python" "$DIAGNOSE_PY" --venv "$VENV_DIR" \
                --llama-cpp "$LLAMA_CPP_DIR" "${extra_args[@]}"
            return $?
        fi
    fi
    local py
    py="$(command -v python3 || command -v python)"
    [ -n "$py" ] || die "no python3 found on PATH"
    "$py" "$DIAGNOSE_PY" --venv "$VENV_DIR" \
        --llama-cpp "$LLAMA_CPP_DIR" "${extra_args[@]}"
}

# ── OS detection ──
if [ -f /etc/os-release ]; then
    . /etc/os-release; DISTRO_ID="$ID"
elif [ "$(uname -s)" = "Darwin" ]; then
    DISTRO_ID="macos"
else
    DISTRO_ID="unknown"
fi
log "OS: $DISTRO_ID ($(uname -srm))"

# ── GPU detection ──
# Sets: GPU_VENDOR=nvidia|amd|intel|none  GPU_NAME  GPU_DRIVER  CUDA_VER
detect_gpu() {
    GPU_VENDOR="none"; GPU_NAME="(no GPU)"; GPU_DRIVER=""; CUDA_VER=""
    [ "$FORCE_CPU" = "1" ] && return 0

    # NVIDIA: nvidia-smi
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 | grep -q .; then
            GPU_VENDOR="nvidia"
            GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
            GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | awk -F. '{print $1}')"
            # Driver → CUDA wheel mapping. abetlen publishes wheels up to
            # cu132 (latest stable as of 2026-09). For drivers on CUDA 13.x
            # (>= 555) we want the cu132 prebuilt (saves the ~5-minute source
            # build). cu130 / cu124 / cu121 / cu118 for older drivers.
            if   [ "${GPU_DRIVER:-0}" -ge 555 ] 2>/dev/null; then CUDA_VER="cu132"
            elif [ "${GPU_DRIVER:-0}" -ge 550 ] 2>/dev/null; then CUDA_VER="cu130"
            elif [ "${GPU_DRIVER:-0}" -ge 525 ] 2>/dev/null; then CUDA_VER="cu124"
            elif [ "${GPU_DRIVER:-0}" -ge 520 ] 2>/dev/null; then CUDA_VER="cu121"
            elif [ "${GPU_DRIVER:-0}" -ge 470 ] 2>/dev/null; then CUDA_VER="cu118"
            else warn "NVIDIA driver $GPU_DRIVER is old — defaulting to CUDA 11.8"
                 CUDA_VER="cu118"
            fi
            return 0
        fi
    fi

    # AMD ROCm
    if command -v rocm-smi >/dev/null 2>&1 || [ -d /opt/rocm ]; then
        GPU_VENDOR="amd"
        GPU_NAME="$(rocm-smi --showproductname 2>/dev/null | grep 'GPU' | head -1 || echo 'AMD GPU (ROCm)')"
        return 0
    fi

    # Intel XPU
    if command -v xpu-smi >/dev/null 2>&1 || [ -d /opt/intel/oneapi ]; then
        GPU_VENDOR="intel"; GPU_NAME="Intel XPU"; return 0
    fi

    # lspci hint
    if command -v lspci >/dev/null 2>&1; then
        if lspci 2>/dev/null | grep -qi 'vga.*nvidia'; then
            warn "NVIDIA GPU detected via lspci but nvidia-smi missing — install your distro's NVIDIA driver."
        fi
    fi
}

detect_gpu
case "$GPU_VENDOR" in
    nvidia) log "GPU: NVIDIA $GPU_NAME — driver $GPU_DRIVER → $CUDA_VER" ;;
    amd)    log "GPU: AMD ROCm — $GPU_NAME" ;;
    intel)  log "GPU: Intel XPU — $GPU_NAME" ;;
    *)      warn "GPU: none — CPU-only mode. Install GPU drivers for 10x+ faster inference." ;;
esac

# ── Find Python 3.12+ ──
pick_python() {
    local cmds=()
    [ -n "$PYTHON_VERSION" ] && cmds+=("python${PYTHON_VERSION}")
    cmds+=(python3.13 python3.12 python3 python)
    for cmd in "${cmds[@]}"; do
        command -v "$cmd" >/dev/null 2>&1 || continue
        local v
        v="$("$cmd" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")"
        case "$v" in 3.1[2-9]) echo "$cmd"; return 0 ;; esac
    done
    return 1
}

PYTHON_CMD="$(pick_python || true)"

# ── Diagnostic-only modes (delegate to Python helper) ──
# These exit before any install/build work, so they can be run cheaply
# from cron, systemd health checks, or CI.
case "$MODE" in
    verify|check)
        # Back-compat: --check still prints a friendly summary, --verify
        # is the strict (exit-code-driven) variant for scripts.
        if [ "$MODE" = "check" ]; then
            # Friendly summary
            run_diagnose --no-service-check
            echo ""
            log "For deeper diagnosis (with service check + JSON), run:"
            log "  bash install.sh --verify"
            log "  python3 scripts/install_diagnose.py --json"
            exit 0
        fi
        # verify: deep diagnostic, exit code reflects health
        #   0 = ok   1 = warnings only   2+ = critical
        run_diagnose --check
        ;;
    repair)
        # Diagnose + autofix. If the diagnostic finds issues that bash
        # can fix (mixed torch, missing llama.cpp CLI), apply the fixes.
        # Anything requiring a venv recreate delegates back to install.
        log "Diagnosing current install..."
        # NB: don't let `set -e` kill us on non-zero — the diagnostic
        # exits 2 to signal critical issues. We capture the rc explicitly.
        # Don't use `run_diagnose || true` here either — that swallows the
        # rc with `true`, making DIAG_RC always 0.
        DIAG_RC=0
        run_diagnose || DIAG_RC=$?
        if [ "$DIAG_RC" = "0" ]; then
            log "✓ install already healthy, nothing to repair."
            exit 0
        fi
        log "Issues found (rc=$DIAG_RC). Applying fixes..."
        # Re-run with --repair to apply; the helper handles torch + llama-cpp
        # fixes in-process. For RECREATE_VENV, it advises recreating the venv
        # and we fall through to the normal install path.
        REPAIR_RC=0
        run_diagnose --repair --no-service-check || REPAIR_RC=$?
        if [ "$REPAIR_RC" -eq 0 ]; then
            log "✓ repair succeeded. Re-running diagnose to verify..."
            DIAG_RC2=0
            run_diagnose || DIAG_RC2=$?
            exit $DIAG_RC2
        fi
        # Repair couldn't autofix everything (likely needs venv recreate).
        # Remove the venv and let the normal install path build a fresh one
        # with the correct GPU-aware wheels.
        log "Repair didn't fully resolve. Recreating venv and re-installing..."
        rm -rf "$VENV_DIR"
        # Fall through to install path
        MODE="install"
        ;;
esac

[ -z "$PYTHON_CMD" ] && die "Python 3.12+ not found. Install:
  Debian/Ubuntu:  sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
  Fedora:         sudo dnf install -y python3.12 python3.12-devel
  Arch/Garuda:    sudo pacman -S python312
  macOS:          brew install python@3.12"
PYTHON_VER="$("$PYTHON_CMD" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
log "Python: $PYTHON_CMD ($PYTHON_VER)"

# ── Bootstrap uv ──
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv..."
    command -v curl >/dev/null 2>&1 && curl -LsSf https://astral.sh/uv/install.sh | sh \
        || { command -v wget >/dev/null 2>&1 && wget -qO- https://astral.sh/uv/install.sh | sh; } \
        || die "Install curl or wget first."
    export PATH="$HOME/.local/bin:$PATH"
fi
log "uv: $(uv --version)"

# ── Create venv ──
if [ "$USE_CONDA" = "1" ]; then
    CONDA_BASE=""
    for cb in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda" "/usr/local/miniconda3"; do
        [ -x "$cb/bin/conda" ] && CONDA_BASE="$cb" && break
    done
    [ -n "$CONDA_BASE" ] || die "conda not found."
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        log "Creating conda env '$CONDA_ENV'..."
        conda create -n "$CONDA_ENV" "python=${PYTHON_VER}" -y
    fi
    conda activate "$CONDA_ENV"
    PYTHON_CMD="$(command -v python)"
    ACTIVATE_HINT="conda activate $CONDA_ENV"
else
    if [ ! -d "$VENV_DIR" ]; then
        log "Creating venv at $VENV_DIR..."
        "$PYTHON_CMD" -m venv "$VENV_DIR" || uv venv "$VENV_DIR" --python "$PYTHON_CMD"
    fi
    PYTHON_CMD="$PWD/$VENV_DIR/bin/python"
    [ -x "$PYTHON_CMD" ] || die "venv broken: $PYTHON_CMD not executable"
    ACTIVATE_HINT="source $VENV_DIR/bin/activate"
fi
log "venv python: $PYTHON_CMD"

# ── pip helper: uv-created venvs have NO pip module (2026-09-25: genorbox1
#    hit "No module named pip" in the llama.cpp requirements step). ──
pip_install() {
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$PYTHON_CMD" "$@"
    else
        "$PYTHON_CMD" -m pip install "$@"
    fi
}

# ── Freeze the torch family after install_torch so later `-e .` / optional
#    installs cannot silently upgrade torch off the CUDA-matched build.
#    (2026-09-25: `uv pip install -e .` replaced torch 2.6.0+cu124 with PyPI
#    2.14.0 (cu130 build) while torchvision/torchaudio stayed +cu124 →
#    ImportError: undefined symbol: ncclCommResume on driver 535.) ──
TORCH_CONSTRAINTS="${VENV_DIR:-.venv}/torch-constraints.txt"
CONSTRAINT_ARGS=()
write_torch_constraints() {
    mkdir -p "$(dirname "$TORCH_CONSTRAINTS")"
    "$PYTHON_CMD" - "$TORCH_CONSTRAINTS" <<'PY' 2>/dev/null || true
import importlib.metadata as m, sys
out = sys.argv[1]
lines = []
for name in ("torch", "torchvision", "torchaudio"):
    try:
        lines.append(f"{name}=={m.version(name)}")
    except m.PackageNotFoundError:
        pass
if lines:
    open(out, "w").write("\n".join(lines) + "\n")
PY
    if [ -s "$TORCH_CONSTRAINTS" ]; then
        CONSTRAINT_ARGS=(-c "$TORCH_CONSTRAINTS")
        log "torch family pinned via $TORCH_CONSTRAINTS"
    fi
}

# ── Install PyTorch with matching GPU ──
install_torch() {
    case "$GPU_VENDOR" in
        nvidia)
            log "Installing PyTorch ($CUDA_VER)..."
            uv pip install --python "$PYTHON_CMD" --reinstall \
                --index-url "https://download.pytorch.org/whl/$CUDA_VER" \
                torch torchvision torchaudio 2>&1 | tail -1 \
            || { warn "Custom index failed — trying default..."; uv pip install --python "$PYTHON_CMD" torch torchvision torchaudio; } ;;
        amd)
            log "Installing PyTorch (ROCm 6.2)..."
            uv pip install --python "$PYTHON_CMD" --reinstall \
                --index-url "https://download.pytorch.org/whl/rocm6.2" \
                torch torchvision torchaudio 2>&1 | tail -1 \
            || { warn "ROCm failed — CPU fallback..."; uv pip install --python "$PYTHON_CMD" torch torchvision torchaudio; } ;;
        intel)
            log "Installing PyTorch (XPU)..."
            uv pip install --python "$PYTHON_CMD" --reinstall \
                --index-url "https://download.pytorch.org/whl/xpu" \
                torch torchvision torchaudio 2>&1 | tail -1 \
            || { warn "XPU failed — CPU fallback..."; uv pip install --python "$PYTHON_CMD" torch torchvision torchaudio; } ;;
        *)
            log "Installing PyTorch (CPU)..."
            uv pip install --python "$PYTHON_CMD" --reinstall \
                --index-url "https://download.pytorch.org/whl/cpu" \
                torch torchvision torchaudio 2>&1 | tail -1 \
            || uv pip install --python "$PYTHON_CMD" torch torchvision torchaudio ;;
    esac
}

# ── Install llama-cpp-python with matching CUDA/ROCm ──
install_gguf() {
    [ "$SKIP_GGUF" = "1" ] && { log "Skipping llama-cpp-python (--no-gguf)."; return 0; }

    case "$GPU_VENDOR" in
        nvidia)
            # abetlen's prebuilt CUDA wheel index (no nvcc/build needed).
            # cu124 for driver 525+, cu121 for 520+, cu118 for 470+.
            log "Installing llama-cpp-python ($CUDA_VER prebuilt wheel — no compilation)..."
            uv pip install --python "$PYTHON_CMD" --reinstall \
                --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/$CUDA_VER/llama-cpp-python/" \
                'llama-cpp-python>=0.3.0' 2>&1 | tail -3 \
            || {
                # Fallback: dougeeai GPU-specific wheel (RTX 3090 = sm_86 Ampere)
                log "abetlen index failed — trying dougeeai prebuilt..."
                uv pip install --python "$PYTHON_CMD" --reinstall \
                    "llama-cpp-python>=0.3.0" \
                    --find-links "https://github.com/dougeeai/llama-cpp-python-wheels/releases" \
                    2>&1 | tail -3 \
                || {
                    # Last resort: source build with CMAKE_ARGS (slow, needs nvcc)
                    warn "Prebuilt wheels failed — building from source (this may take 5-15 min)..."
                    CMAKE_ARGS="-DGGML_CUDA=on" \
                        uv pip install --python "$PYTHON_CMD" --reinstall \
                        'llama-cpp-python>=0.3.0' --no-binary llama-cpp-python \
                    || warn "llama-cpp-python CUDA build failed — GGUF inference unavailable."
                }
            } ;;
        *)
            log "Installing llama-cpp-python (CPU — install NVIDIA driver for GPU wheel)..."
            uv pip install --python "$PYTHON_CMD" --reinstall 'llama-cpp-python>=0.3.0' ;;
    esac
}

# ── Build llama.cpp CLI tools (convert_hf_to_gguf.py + llama-quantize) ──
# Required by the GGUF export endpoint. Skipped via --no-llama-cpp.
# Use --llama-cpp-only to JUST build this (skip torch, llama-cpp-python, base pkgs).
install_llama_cpp_cli() {
    [ "$SKIP_LLAMA_CPP" = "1" ] && { log "Skipping llama.cpp CLI build (--no-llama-cpp)."; return 0; }

    local convert_script="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
    local quantize_bin="$LLAMA_CPP_DIR/build/bin/llama-quantize"

    if [ -f "$convert_script" ] && [ -x "$quantize_bin" ]; then
        log "llama.cpp CLI present at $LLAMA_CPP_DIR (skipping build)."
        return 0
    fi

    log "Building llama.cpp CLI at $LLAMA_CPP_DIR (needed by export endpoint)..."
    command -v cmake >/dev/null 2>&1 || die "cmake not found. Install build-essential + cmake."
    command -v git  >/dev/null 2>&1 || die "git not found."
    if [ ! -d "$LLAMA_CPP_DIR" ]; then
        git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR" \
            || die "git clone llama.cpp failed"
    fi
    pip_install --quiet \
        -r "$LLAMA_CPP_DIR/requirements/requirements-convert_hf_to_gguf.txt" 2>&1 | tail -3 \
        || warn "convert_hf_to_gguf pip deps install failed — conversion may not work."
    cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" 2>&1 | tail -2 \
        || die "cmake configure failed."
    cmake --build "$LLAMA_CPP_DIR/build" --config Release -j 2>&1 | tail -3 \
        || die "cmake build failed."

    if [ ! -x "$quantize_bin" ]; then
        die "llama-quantize not built. Check the cmake output above."
    fi
    log "llama-quantize: present at $quantize_bin"
}

# ── Install everything ──
if [ "$LLAMA_CPP_ONLY" = "1" ]; then
    install_llama_cpp_cli
    exit 0
fi
install_torch
write_torch_constraints
install_gguf
log "Installing base packages from pyproject.toml..."
uv pip install --python "$PYTHON_CMD" "${CONSTRAINT_ARGS[@]}" -e .
mkdir -p data
install_llama_cpp_cli

# ── Install optional but recommended packages ──
# These are used by advanced features (abliteration, GPTQ export, Unsloth).
# They're installed silently — if they fail, the app still works.
install_optional_packages() {
    log "Installing optional packages (unsloth, gptq extra, numpy, scipy)..."
    # numpy and scipy are needed for abliteration (always useful)
    uv pip install --python "$PYTHON_CMD" "numpy>=1.24.0" "scipy>=1.10.0" 2>&1 | tail -1 \
        || warn "numpy/scipy install failed — abliteration may not work."
    # unsloth for faster training
    uv pip install --python "$PYTHON_CMD" "${CONSTRAINT_ARGS[@]}" "unsloth>=2024.10.0" 2>&1 | tail -1 \
        || warn "unsloth install failed — will use standard training."
    # GPTQ export (gptqmodel) + Transformers GPTQ load (optimum) via pyproject extra
    uv pip install --python "$PYTHON_CMD" "${CONSTRAINT_ARGS[@]}" -e '.[gptq]' 2>&1 | tail -1 \
        || warn "gptq extra install failed — try: uv pip install -e '.[gptq]' (gptqmodel + optimum)."
}
install_optional_packages

# ── Verify ──
log "Verifying..."
"$PYTHON_CMD" -c "import fastapi, jinja2; print(f'  fastapi={fastapi.__version__} jinja2={jinja2.__version__}')" || die "Install failed."
"$PYTHON_CMD" -c "
import torch; a='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'  torch: {a}', flush=True)
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB)', flush=True)
" 2>&1 | sed 's/^/[install]   /'
[ "$SKIP_GGUF" = "0" ] && "$PYTHON_CMD" -c "import llama_cpp; print(f'  llama-cpp-python: {llama_cpp.__version__}')" 2>/dev/null | sed 's/^/[install]   /' || true
[ -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ] && log "  llama.cpp CLI: $LLAMA_CPP_DIR" || true

# ── Post-install health check (autodetect broken installs) ──
# The deep diagnostic catches issues the surface checks miss
# (mixed torch family, torchaudio+libc10_cuda.so mismatches, etc.).
# On a fresh install this should be a no-op; on a re-install into an
# existing venv it surfaces problems that need --repair.
log "Running deep health check (scripts/install_diagnose.py)..."
DIAG_RC=0
run_diagnose || DIAG_RC=$?
if [ "$DIAG_RC" = "0" ]; then
    log "  ✓ deep health check passed."
elif [ "${AUTO_REPAIR:-0}" = "1" ] || [ "${FTS_AUTO_REPAIR:-0}" = "1" ]; then
    log "  Deep health check found issues (rc=$DIAG_RC). Auto-repair enabled — applying fixes..."
    REPAIR_RC=0
    run_diagnose --repair --no-service-check || REPAIR_RC=$?
    if [ "$REPAIR_RC" = "0" ]; then
        log "  Re-verifying after repair..."
        DIAG_RC2=0
        run_diagnose || DIAG_RC2=$?
        if [ "$DIAG_RC2" = "0" ]; then
            log "  ✓ install now fully healthy."
        else
            warn "  auto-repair applied but the system is still unhealthy (rc=$DIAG_RC2)."
            warn "  Re-run with --verify for details."
        fi
    else
        warn "  auto-repair did not fully resolve (rc=$REPAIR_RC)."
        warn "  Try: bash install.sh --repair"
    fi
else
    warn "Deep health check found issues. Run  bash install.sh --repair  to autofix,"
    warn "or  bash install.sh --verify  to see details. Quick fix:"
    warn "  bash install.sh --repair"
    warn "Or set FTS_AUTO_REPAIR=1 to make install.sh auto-fix without prompting."
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
log "✓ Install complete."
echo ""
echo "Activate:"
echo "  bash/zsh:  $ACTIVATE_HINT"
case "$ACTIVATE_HINT" in
    *"conda activate"*) echo "  fish:      conda activate $CONDA_ENV" ;;
    *"source "*)        echo "  fish:      source $VENV_DIR/bin/activate.fish" ;;
esac
echo ""
echo "Run:           $PYTHON_CMD -m finetune_studio --help"
echo "WebUI:         $PYTHON_CMD -m uvicorn finetune_studio.webui.app:app --host 0.0.0.0 --port 7860"
echo "Verify:        bash install.sh --check"
echo "═══════════════════════════════════════════════════════════════"