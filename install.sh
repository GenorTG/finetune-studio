#!/usr/bin/env bash
# Finetune Studio — GPU-aware installer.
#
# Default:
#   1. Detects the accelerator (NVIDIA CUDA / AMD ROCm / Intel XPU / Apple Metal;
#      CPU only when NO GPU of any vendor is found). Detection + every wheel/index/
#      CMake choice lives in scripts/accel_plan.py (shared with update.sh and the
#      Windows installers) -- never hard-code a CUDA/ROCm tag here.
#   2. Python 3.12+ required
#   3. Creates .venv + installs the newest PyTorch the driver/GPU supports
#   4. Installs llama-cpp-python with GPU offload (prebuilt wheel, else source
#      build with GGML_CUDA / GGML_HIP / GGML_SYCL / GGML_VULKAN / GGML_METAL)
#   5. Installs everything else from pyproject.toml + bitsandbytes (+ unsloth when
#      a compatible release exists)
#   6. Builds the llama.cpp CLI (.llama.cpp) with the same GPU backend
#
# Env: FTS_FORCE_VENDOR=nvidia|amd|intel|apple|none  FTS_UNSLOTH=auto|1|0
#      FTS_ACCEL_FIXTURE=<json>  (fake hardware for tests)
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
#   --gpu VENDOR    override detection: nvidia | amd | intel | apple | none
#   --plan          print the detected accelerator + install plan, change nothing
#   --rebuild-llama-cpp  rebuild the llama.cpp CLI even if present
#   --no-gguf       skip llama-cpp-python entirely
#   --no-llama-cpp  skip building the llama.cpp CLI (.llama.cpp/)
#   --llama-cpp-only  build ONLY the llama.cpp CLI (skip torch + packages)
#   --help          show usage

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PYTHON_VERSION="${PYTHON_VERSION:-}"
USE_CONDA="${USE_CONDA:-0}"
CONDA_ENV="${CONDA_ENV:-chris-ai}"
VENV_DIR="${VENV_DIR:-.venv}"
# llama.cpp lives inside the project, NOT in $HOME. Boxed-in so the install
# is reproducible and self-contained — nothing references external paths.
log()  { if [ "${PLAN_ONLY:-0}" = "1" ]; then echo "[install] $*" >&2; else echo "[install] $*"; fi; }  # --plan: stdout is the plan JSON only
warn() { echo "[install] WARN: $*" >&2; }
die()  { echo "[install] ERROR: $*" >&2; exit 1; }

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
FORCE_GPU=""
PLAN_ONLY=0
REBUILD_LLAMA=0
SKIP_GGUF=0
SKIP_LLAMA_CPP=0
LLAMA_CPP_ONLY=0

while [ "$#" -gt 0 ]; do
    arg="$1"; shift
    case "$arg" in
        --gpu)            FORCE_GPU="${1:-}"; [ -n "$FORCE_GPU" ] || die "--gpu needs a vendor"; shift ;;
        --gpu=*)          FORCE_GPU="${arg#--gpu=}" ;;
        --plan)           PLAN_ONLY=1 ;;
        --rebuild-llama-cpp) REBUILD_LLAMA=1 ;;
        --check)          MODE="check" ;;
        --repair)         MODE="repair" ;;
        --verify)         MODE="verify" ;;
        --auto-repair)    AUTO_REPAIR=1 ;;
        --cpu)            FORCE_CPU=1 ;;
        --no-gguf)        SKIP_GGUF=1 ;;
        --no-llama-cpp)   SKIP_LLAMA_CPP=1 ;;
        --llama-cpp-only) LLAMA_CPP_ONLY=1 ;;
        --help|-h) sed -n '2,40p' "$0" | sed 's/^# *//'; exit 0 ;;
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

# ── GPU detection (delegated to scripts/accel_plan.py) ──
# One implementation for install.sh / update.sh / install.ps1 / install.bat /
# install_diagnose.py --repair: it probes nvidia-smi, rocm-smi/rocminfo//opt/rocm,
# xpu-smi/sycl-ls/clinfo, sysfs PCI and lspci, so a GPU with a missing driver
# still gets GPU wheels plus an actionable hint instead of a silent CPU install.
ACCEL_PY="$PROJECT_ROOT/scripts/accel_plan.py"
ACCEL_FLAGS=()
[ "$FORCE_CPU" = "1" ] && ACCEL_FLAGS+=(--cpu)
[ -n "$FORCE_GPU" ] && ACCEL_FLAGS+=(--gpu "$FORCE_GPU")
accel_py() {  # any python3 works: the planner is stdlib-only
    local py="${PYTHON_CMD:-}"
    [ -n "$py" ] || py="$(command -v python3 || command -v python || true)"
    [ -n "$py" ] || die "no python3 found on PATH"
    echo "$py"
}
accel() {  # accel <subcommand> [args...]  (venv python once the venv exists)
    local sub="$1"; shift
    "$(accel_py)" "$ACCEL_PY" "$sub" ${ACCEL_FLAGS[@]+"${ACCEL_FLAGS[@]}"} "$@"
}

# Sets: GPU_VENDOR GPU_NAME GPU_DRIVER CUDA_VER TORCH_TAG TORCH_INDEX LLAMA_BACKEND
#       LLAMA_CMAKE_ARGS LLAMA_WHEEL_INDEX NVCC DRIVER_READY PLAN_WARNINGS ...
load_plan() {
    local vars
    vars="$(accel plan --shell)" || die "accelerator detection failed (scripts/accel_plan.py)"
    eval "$vars"
    case "$GPU_VENDOR" in
        nvidia) log "GPU: NVIDIA $GPU_NAME — driver $GPU_DRIVER (CUDA ${CUDA_MAX:-?}) → torch $TORCH_TAG, llama.cpp $LLAMA_BACKEND" ;;
        amd)    log "GPU: AMD $GPU_NAME — ROCm ${ROCM_VERSION:-n/a} → torch $TORCH_TAG, llama.cpp $LLAMA_BACKEND" ;;
        intel)  log "GPU: Intel $GPU_NAME → torch xpu, llama.cpp $LLAMA_BACKEND" ;;
        apple)  log "GPU: Apple Silicon → torch mps, llama.cpp $LLAMA_BACKEND" ;;
        *)      log "GPU: none — CPU wheels" ;;
    esac
    if [ -n "$PLAN_WARNINGS" ]; then
        while IFS= read -r line; do [ -n "$line" ] && warn "$line"; done <<<"$PLAN_WARNINGS"
    fi
}

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

# ── --plan: show what would be installed, change nothing ──
if [ "$PLAN_ONLY" = "1" ]; then
    accel plan
    exit 0
fi

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
        DIAG_RC=0
        run_diagnose --check || DIAG_RC=$?
        exit "$DIAG_RC"
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

load_plan
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

# ── torch family pin: $VENV_DIR/torch-constraints.txt is written right after the
#    GPU torch install (accel_plan.py `constraints`) and passed as `-c` to every
#    later resolve so `-e .`, bitsandbytes, unsloth cannot swap torch off its GPU
#    build (2026-09-25: `uv pip install -e .` replaced torch 2.6.0+cu124 with the
#    PyPI cu130 build -> undefined symbol ncclCommResume on driver 535). ──
TORCH_CONSTRAINTS="${VENV_DIR:-.venv}/torch-constraints.txt"
CONSTRAINT_ARGS=()
refresh_constraints() {
    if accel constraints --venv "$VENV_DIR" >/dev/null 2>&1 && [ -s "$TORCH_CONSTRAINTS" ]; then
        CONSTRAINT_ARGS=(-c "$TORCH_CONSTRAINTS")
        log "torch family pinned via $TORCH_CONSTRAINTS"
    fi
}

# ── Install PyTorch (newest index the driver + GPU support; never CPU on a GPU host) ──
install_torch() {
    accel install torch --venv "$VENV_DIR" \
        || die "PyTorch install failed for $GPU_VENDOR ($TORCH_TAG). See the messages above; \
re-run after fixing the driver, or force CPU explicitly with --cpu."
}

# ── llama-cpp-python: GPU prebuilt wheel, else source build with the right GGML_* flags ──
install_gguf() {
    [ "$SKIP_GGUF" = "1" ] && { log "Skipping llama-cpp-python (--no-gguf)."; return 0; }
    accel install llama-cpp-python --venv "$VENV_DIR" \
        || warn "llama-cpp-python install failed — GGUF inference unavailable."
}

# ── llama.cpp CLI (convert_hf_to_gguf.py + llama-quantize + llama-cli), same backend ──
# Required by the GGUF export endpoint. Skipped via --no-llama-cpp.
# --llama-cpp-only builds just this (skips torch, llama-cpp-python, base pkgs).
install_llama_cpp_cli() {
    [ "$SKIP_LLAMA_CPP" = "1" ] && { log "Skipping llama.cpp CLI build (--no-llama-cpp)."; return 0; }
    local args=(--dir "$LLAMA_CPP_DIR" --venv "$VENV_DIR")
    [ "$LLAMA_CPP_ONLY" = "1" ] && args+=(--only)
    [ "$REBUILD_LLAMA" = "1" ] && args+=(--rebuild)
    accel build-llama-cli "${args[@]}" || die "llama.cpp CLI build failed (see messages above)."
}

# ── Install everything ──
if [ "$LLAMA_CPP_ONLY" = "1" ]; then
    refresh_constraints
    install_llama_cpp_cli
    exit 0
fi
install_torch
refresh_constraints
install_gguf
log "Installing base packages from pyproject.toml..."
# [parsers] is folded into the base install (not an optional extra): file
# upload/parse is core functionality, not an advanced feature like unsloth
# or abliteration. Without pypdf/python-docx/openpyxl/xlrd/python-pptx/
# beautifulsoup4/striprtf/Pillow, PDF/DOCX/XLSX/PPTX/HTML/RTF sources
# silently fail to parse on a fresh install.
uv pip install --python "$PYTHON_CMD" ${CONSTRAINT_ARGS[@]+"${CONSTRAINT_ARGS[@]}"} -e '.[parsers]'
mkdir -p data
install_llama_cpp_cli

# ── Optional-but-recommended packages (the app still works if these fail) ──
install_optional_packages() {
    log "Installing optional packages (bitsandbytes, numpy, scipy, unsloth)..."
    # bitsandbytes ships one wheel with CUDA, ROCm and XPU backends
    accel install bitsandbytes --venv "$VENV_DIR" \
        || warn "bitsandbytes install failed — 4-bit/8-bit loading unavailable."
    # numpy and scipy are needed for abliteration (always useful)
    uv pip install --python "$PYTHON_CMD" ${CONSTRAINT_ARGS[@]+"${CONSTRAINT_ARGS[@]}"} "numpy>=1.24.0" "scipy>=1.10.0" 2>&1 | tail -1 \
        || warn "numpy/scipy install failed — abliteration may not work."
    # unsloth: NVIDIA only, installed ONLY when a release resolves against the
    # installed torch/transformers/trl (else skipped with a note, never a downgrade)
    accel install unsloth --venv "$VENV_DIR" \
        || warn "unsloth install failed — will use standard training."
}
install_optional_packages

# ── Verify ──
log "Verifying..."
"$PYTHON_CMD" -c "import fastapi, jinja2; print(f'  fastapi={fastapi.__version__} jinja2={jinja2.__version__}')" || die "Install failed."
"$PYTHON_CMD" -c "
import torch
v = torch.__version__
if torch.cuda.is_available():
    kind = 'rocm/hip' if getattr(torch.version, 'hip', None) else 'cuda'
    print(f'  torch {v}: {kind}  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB)', flush=True)
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    print(f'  torch {v}: xpu  GPU: {torch.xpu.get_device_name(0)}', flush=True)
elif torch.backends.mps.is_available():
    print(f'  torch {v}: mps', flush=True)
else:
    print(f'  torch {v}: cpu (no usable accelerator at runtime)', flush=True)
" 2>&1 | sed 's/^/[install]   /'
if [ "$SKIP_GGUF" = "0" ]; then
    "$PYTHON_CMD" -c "import llama_cpp; print(f'  llama-cpp-python {llama_cpp.__version__}: gpu_offload={llama_cpp.llama_supports_gpu_offload()}')" 2>/dev/null | sed 's/^/[install]   /' || true
fi
"$PYTHON_CMD" -c "import bitsandbytes as b; print(f'  bitsandbytes {b.__version__}')" 2>/dev/null | sed 's/^/[install]   /' || true
[ -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ] && log "  llama.cpp CLI: $LLAMA_CPP_DIR (backend: $(cat "$LLAMA_CPP_DIR/build/.fts-backend" 2>/dev/null || echo unknown))" || true
uv pip check --python "$PYTHON_CMD" 2>&1 | sed 's/^/[install]   check: /' | head -8 || true

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