"""Install diagnostics + auto-repair for Finetune Studio.

The shell install scripts (`install.sh`, `update.sh`) call this module to:

  diagnose(venv_dir, llama_cpp_dir)
      → list of Issue(code, detail, suggested_fix)
      → exit-code semantics: 0 healthy, 1 minor, 2 venv broken, 3 GPU broken

  repair(issues, venv_dir, llama_cpp_dir, log=print)
      → applies fixes in dependency order, returns (ok, summary)

The functions are pure (no side effects on import), use subprocess for
torch/llama-cpp introspection so they don't load the broken modules in
the test process, and are designed to be unit-tested with subprocess
mocks.

Detected failure modes (the ones we've hit in production):
  - venv python missing / non-executable               → RECREATE_VENV
  - venv python broken (sys/os import fails)           → RECREATE_VENV
  - mixed torch installs (torch+cpu + torchaudio+cu130)  → REINSTALL_TORCH
  - torch+cpu when GPU available                      → REINSTALL_TORCH
  - torch.cuda.is_available() == False with GPU       → REINSTALL_TORCH
  - torch.cuda can allocate tensor fails              → REINSTALL_TORCH
  - torchaudio import fails (libc10_cuda.so missing)   → REINSTALL_TORCH
  - torch built for the wrong backend (cpu/cuda/rocm/xpu vs the GPU) → REINSTALL_TORCH
  - llama-cpp-python not importable / no GPU offload   → REINSTALL_LLAMA_CPP
  - llama.cpp CLI missing or built CPU-only on a GPU   → BUILD_LLAMA_CPP_CLI
  - bitsandbytes missing on a GPU host                 → INSTALL_BITSANDBYTES
  - pyproject deps missing                           → PIP_INSTALL_EDITABLE
  - GPU hardware but no driver / toolchain (manual)    → INSTALL_GPU_DRIVER / INSTALL_GPU_TOOLCHAIN
  - CUDA toolkit missing on GPU host                  → INSTALL_CUDA_TOOLKIT
  - systemd unit missing on a host that should run it → INSTALL_SERVICE

Detection + every wheel/index/CMake decision lives in accel_plan.py (shared with
the installers); this module only compares what is installed against that plan.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import accel_plan as ap
from accel_plan import (
    GpuInfo,
)

log = logging.getLogger(__name__)

# Failures a probe/repair subprocess can legitimately produce (spawn failure,
# timeout, unreadable/odd output). Anything else is a bug and should surface.
_STEP_ERRORS = (subprocess.SubprocessError, OSError)
_PROBE_ERRORS = (*_STEP_ERRORS, ValueError, KeyError, IndexError)

# ── Issue taxonomy ──────────────────────────────────────────────────────

RECREATE_VENV        = "recreate-venv"
REINSTALL_TORCH      = "reinstall-torch"
REINSTALL_LLAMA_CPP  = "reinstall-llama-cpp"
BUILD_LLAMA_CPP_CLI  = "build-llama-cpp-cli"
PIP_INSTALL_EDITABLE = "pip-install-editable"
INSTALL_CUDA_TOOLKIT = "install-cuda-toolkit"
INSTALL_SERVICE      = "install-service"
INSTALL_BITSANDBYTES = "install-bitsandbytes"
INSTALL_GPU_DRIVER   = "install-gpu-driver"        # manual: needs sudo / reboot
INSTALL_GPU_TOOLCHAIN = "install-gpu-toolchain"    # manual: nvcc/hipcc/icpx/vulkan sdk

# Issue codes no script can fix without root: repair() reports them, never "fails" on them.
MANUAL_CODES = frozenset({INSTALL_GPU_DRIVER, INSTALL_GPU_TOOLCHAIN})


@dataclass(frozen=True)
class Issue:
    code: str           # RECREATE_VENV, REINSTALL_TORCH, ...
    detail: str         # human-readable explanation
    suggested_fix: str  # human-readable actionable fix
    severity: int = 1   # 0=info, 1=warn, 2=error, 3=critical (recreate needed)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Issue:
        return cls(**d)


# ── Diagnostic helpers ──────────────────────────────────────────────────

@dataclass
class VenvInfo:
    path: Path
    python: Path | None
    py_version: str | None
    torch_version: str | None
    torch_has_cuda: bool | None
    torchaudio_version: str | None
    torchaudio_importable: bool | None
    transformers_version: str | None
    llama_cpp_version: str | None
    peft_version: str | None
    trl_version: str | None
    has_fastapi: bool
    has_jinja2: bool
    has_datasets: bool
    has_accelerate: bool
    has_safetensors: bool
    has_huggingface_hub: bool
    # Build flavor + runtime availability of the installed torch (see _TORCH_PROBE).
    torch_kind: str | None = None        # "cuda" | "hip" | "xpu" | "mps" | "cpu"
    torch_xpu: bool | None = None
    torch_mps: bool | None = None
    missing_parsers: tuple[str, ...] = ()   # modules of the [parsers] extra that do not import

    @property
    def exists(self) -> bool:
        return self.path.exists() and self.python is not None

    @property
    def healthy(self) -> bool:
        return (
            self.python is not None
            and self.py_version is not None
            and self.has_fastapi and self.has_jinja2
            and self.torch_version is not None
            and self.torchaudio_importable in (True, None)  # None = not installed (the app does not use it)
            and self.torch_has_cuda in (True, None)  # None = no GPU expected
        )


# Build flavor of torch: version tags alone lie (a "+cu132" tag on a CPU-only
# machine, ROCm builds report cuda=True). torch.version.* tells the truth.
_TORCH_PROBE = (
    "import torch, json\n"
    "x = getattr(torch, 'xpu', None)\n"
    "print(json.dumps({'v': torch.__version__, 'cuda': torch.cuda.is_available(),\n"
    "  'build_cuda': torch.version.cuda, 'build_hip': getattr(torch.version, 'hip', None),\n"
    "  'xpu': bool(x and x.is_available()), 'build_xpu': '+xpu' in torch.__version__,\n"
    "  'mps': bool(torch.backends.mps.is_available())}))"
)


_TORCHAUDIO_PROBE = (
    "import importlib.util as u\n"
    "if u.find_spec('torchaudio') is None:\n"
    "    print('ABSENT')\n"
    "else:\n"
    "    import torchaudio; print(torchaudio.__version__)"
)


def _torch_kind(j: dict) -> str:
    """cuda | hip | xpu | mps | cpu from the probe payload (older payloads: version tag)."""
    if j.get("build_hip"):
        return "hip"
    if j.get("build_cuda"):
        return "cuda"
    if j.get("build_xpu") or j.get("xpu"):
        return "xpu"
    if j.get("mps"):
        return "mps"
    if "build_cuda" in j:
        return "cpu"
    v = str(j.get("v", ""))  # legacy payload {'v','cuda'}
    return "cuda" if (j.get("cuda") or "+cu" in v) else "cpu"


def _run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False,
        env={**os.environ, "PYTHONPATH": "", "VIRTUAL_ENV": ""},
    )


_DEP_MODULES = ("fastapi", "jinja2", "datasets", "accelerate", "safetensors", "huggingface_hub")
# The `[parsers]` extra: import name -> what stops working without it. Their absence never breaks the app
# (each parser degrades or refuses its format), which is exactly why nothing noticed that a host had been
# missing xlrd / python-pptx / beautifulsoup4 / striprtf: .xls, .pptx, .rtf failed and HTML parsed with a regex.
_PARSER_MODULES = {
    "pypdf": ".pdf", "docx": ".docx", "olefile": "legacy .doc", "openpyxl": ".xlsx", "xlrd": ".xls",
    "pptx": ".pptx", "bs4": "HTML (regex fallback)", "striprtf": ".rtf", "PIL": "images/OCR input",
}
_DEPS_PROBE = (
    "import importlib, json, sys\n"
    f"mods = {list(_DEP_MODULES)!r}\n"
    f"parsers = {list(_PARSER_MODULES)!r}\n"
    "def absent(names):\n"
    "    out = []\n"
    "    for m in names:\n"
    "        try:\n"
    "            importlib.import_module(m)\n"
    "        except Exception:\n"
    "            out.append(m)\n"
    "    return out\n"
    "missing = absent(mods)\n"
    "print(json.dumps({'missing': missing, 'missing_parsers': absent(parsers)}))\n"
    "sys.exit(1 if missing else 0)\n"
)


def _parse_missing_deps(r: subprocess.CompletedProcess) -> set[str]:
    """Modules the dependency probe reported absent; everything if it never reported."""
    if r.returncode == 0:
        return set()
    try:
        report = json.loads((r.stdout or "").strip().splitlines()[-1])
        return {m for m in report["missing"] if m in _DEP_MODULES}
    except _PROBE_ERRORS:
        return set(_DEP_MODULES)  # probe crashed before reporting: claim nothing


def _parse_missing_parsers(r: subprocess.CompletedProcess) -> tuple[str, ...]:
    """Parser modules the probe reported absent; empty when it never said (claim nothing)."""
    try:
        report = json.loads((r.stdout or "").strip().splitlines()[-1])
        return tuple(m for m in report.get("missing_parsers", []) if m in _PARSER_MODULES)
    except (*_PROBE_ERRORS, AttributeError):
        return ()


def inspect_venv(venv_dir: Path) -> VenvInfo:
    """Read installed package versions + importability WITHOUT loading
    the host process's site-packages. All checks go through the venv's
    own python via -c, so a broken venv doesn't poison this process."""
    py = venv_dir / "bin" / "python"
    py = py if py.exists() else venv_dir / "Scripts" / "python.exe"
    info = VenvInfo(
        path=venv_dir, python=py if py.exists() else None,
        py_version=None, torch_version=None, torch_has_cuda=None,
        torchaudio_version=None, torchaudio_importable=None,
        transformers_version=None, llama_cpp_version=None,
        peft_version=None, trl_version=None,
        has_fastapi=False, has_jinja2=False, has_datasets=False,
        has_accelerate=False, has_safetensors=False, has_huggingface_hub=False,
    )
    if not info.python:
        return info

    # Basic python health
    try:
        r = _run([str(info.python), "-c",
                  ("import sys, json; print(json.dumps("
                  "{'py': sys.version.split()[0], 'executable': sys.executable}"
                  "))")], timeout=15)
        if r.returncode != 0:
            return info  # venv python broken
        j = json.loads(r.stdout.strip().splitlines()[-1])
        info.py_version = j["py"]
    except _PROBE_ERRORS as exc:
        log.debug("venv python probe failed: %s", exc)
        return info

    # Key-dependency import sanity. ONE subprocess, but each package gets its
    # own verdict: the probe exits 0 iff every import works, otherwise it
    # prints {"missing": [...]} so one absent package cannot mark the other
    # five as missing too.
    try:
        r = _run([str(info.python), "-c", _DEPS_PROBE], timeout=30)
        missing = _parse_missing_deps(r)
        info.missing_parsers = _parse_missing_parsers(r)
    except _PROBE_ERRORS as exc:
        log.debug("dependency probe failed: %s", exc)
        missing = set(_DEP_MODULES)
    info.has_fastapi = "fastapi" not in missing
    info.has_jinja2 = "jinja2" not in missing
    info.has_datasets = "datasets" not in missing
    info.has_accelerate = "accelerate" not in missing
    info.has_safetensors = "safetensors" not in missing
    info.has_huggingface_hub = "huggingface_hub" not in missing

    # torch (may fail on mixed installs — capture separately)
    try:
        r = _run([str(info.python), "-c", _TORCH_PROBE], timeout=45)
        if r.returncode == 0 and r.stdout.strip():
            j = json.loads(r.stdout.strip().splitlines()[-1])
            info.torch_version = j["v"]
            info.torch_has_cuda = bool(j["cuda"])  # True for ROCm builds too (HIP aliases torch.cuda)
            info.torch_kind = _torch_kind(j)
            info.torch_xpu = bool(j.get("xpu"))
            info.torch_mps = bool(j.get("mps"))
    except _PROBE_ERRORS as exc:
        log.debug("torch probe failed: %s", exc)

    # torchaudio — separate so a broken torchaudio doesn't mask torch status.
    # Absent is fine (the app never imports it and it is frozen at 2.11); only an
    # INSTALLED-but-unimportable torchaudio is the mixed-install canary.
    try:
        r = _run([str(info.python), "-c", _TORCHAUDIO_PROBE], timeout=30)
        out = (r.stdout.strip().splitlines() or [""])[-1]
        if r.returncode == 0 and out == "ABSENT":
            info.torchaudio_importable = None
        elif r.returncode == 0:
            info.torchaudio_version = out
            info.torchaudio_importable = True
        else:
            info.torchaudio_importable = False
    except _PROBE_ERRORS as exc:
        log.debug("torchaudio probe failed: %s", exc)
        info.torchaudio_importable = False

    # transformers / llama-cpp / peft / trl — optional but expected
    for pkg, attr in (
        ("transformers", "transformers_version"),
        ("llama_cpp", "llama_cpp_version"),
        ("peft", "peft_version"),
        ("trl", "trl_version"),
    ):
        try:
            r = _run([str(info.python), "-c",
                      f"import {pkg}; print({pkg}.__version__)"], timeout=15)
            if r.returncode == 0:
                setattr(info, attr, r.stdout.strip().splitlines()[-1])
        except _PROBE_ERRORS as exc:
            log.debug("%s probe failed: %s", pkg, exc)

    return info


def inspect_llama_cpp(llama_cpp_dir: Path) -> dict:
    """Check whether llama.cpp CLI tools are built and reachable."""
    quantize = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    cli = llama_cpp_dir / "build" / "bin" / "llama-cli"
    convert = llama_cpp_dir / "convert_hf_to_gguf.py"
    return {
        "dir": str(llama_cpp_dir),
        "quantize_exists": quantize.exists() and os.access(quantize, os.X_OK),
        "cli_exists": cli.exists() and os.access(cli, os.X_OK),
        "backend": ap.detect_llama_cli_backend(llama_cpp_dir),  # cuda|hip|sycl|vulkan|metal|cpu|""
        "convert_exists": convert.exists(),
        "quantize_path": str(quantize) if quantize.exists() else "",
        "convert_path": str(convert) if convert.exists() else "",
    }


def inspect_service() -> dict:
    """Check whether finetune-studio systemd unit is installed + active."""
    out = {
        "systemd_available": bool(shutil.which("systemctl")),
        "user_unit_path": str(Path.home() / ".config/systemd/user/finetune-studio.service"),
        "user_unit_installed": False,
        "user_service_active": False,
        "system_unit_installed": False,
        "lingering_enabled": False,
    }
    if not out["systemd_available"]:
        return out
    unit = Path(out["user_unit_path"])
    out["user_unit_installed"] = unit.exists()
    if out["user_unit_installed"]:
        try:
            r = subprocess.run(
                ["systemctl", "--user", "is-active", "finetune-studio"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            out["user_service_active"] = (r.stdout.strip() == "active")
        except _STEP_ERRORS as exc:
            log.debug("systemctl is-active failed: %s", exc)
    try:
        r = subprocess.run(
            ["loginctl", "show-user", os.environ.get("USER", "root"),
             "-p", "Linger"], capture_output=True, text=True, timeout=5, check=False,
        )
        out["lingering_enabled"] = ("Linger=yes" in r.stdout)
    except _STEP_ERRORS as exc:
        log.debug("loginctl show-user failed: %s", exc)
    return out


def probe_llama_offload(py: Path) -> bool | None:
    """True/False: does the installed llama-cpp-python offload to a GPU backend? None: unknown."""
    try:
        r = _run([str(py), "-c", ap.LLAMA_PROBE], timeout=90)
    except _PROBE_ERRORS as exc:
        log.debug("llama offload probe failed: %s", exc)
        return None
    if r.returncode == 0:
        return True
    return False if r.returncode == 4 else None  # 4 = imported fine, no GPU backend


def probe_bitsandbytes(py: Path) -> str | None:
    try:
        r = _run([str(py), "-c", "import importlib.metadata as m; print(m.version('bitsandbytes'))"], timeout=30)
    except _PROBE_ERRORS as exc:
        log.debug("bitsandbytes probe failed: %s", exc)
        return None
    return r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else None


def _llama_build_marker(venv: Path) -> dict:
    """Written by accel_plan after each llama-cpp-python install: how it was built."""
    try:
        return json.loads((venv / ap.LLAMA_MARKER_NAME).read_text())
    except (OSError, ValueError):
        return {}


# ── Per-area checks (each returns its own issues; diagnose() just concatenates) ──

def _torch_reinstall_hint(venv: VenvInfo, plan: ap.Plan) -> str:
    if plan.torch_index:
        return (f"reinstall torch:  uv pip install --python {venv.python} --reinstall-package torch "
                f"--reinstall-package torchvision --index-url {plan.torch_index} torch torchvision"
                f"   (or:  bash install.sh --repair)")
    return f"reinstall torch:  uv pip install --python {venv.python} --reinstall torch torchvision"


def _check_torch(venv: VenvInfo, gpu: GpuInfo, plan: ap.Plan, force_cpu: bool) -> list[Issue]:
    out: list[Issue] = []
    if venv.torch_version is None:
        out.append(Issue(REINSTALL_TORCH, "torch not importable",
                         f"reinstall torch with the matching {plan.torch_backend} wheels "
                         f"(gpu={gpu.vendor}, index={plan.torch_tag or 'default'})", severity=3))
        return out
    kind = venv.torch_kind or ("cuda" if "+cu" in venv.torch_version else "cpu")
    want = plan.torch_backend
    fix = _torch_reinstall_hint(venv, plan)

    if gpu.vendor in ("nvidia", "amd", "intel", "apple"):
        if kind == "cpu" or "+cpu" in venv.torch_version:
            out.append(Issue(
                REINSTALL_TORCH,
                f"torch {venv.torch_version} is a CPU-only build but a {gpu.vendor.upper()} GPU "
                f"({gpu.name}) is present and needs {plan.torch_tag or 'the default'} wheels", fix, severity=2))
        elif kind != want:
            out.append(Issue(
                REINSTALL_TORCH,
                f"torch {venv.torch_version} is a {kind} build but this host's GPU ({gpu.name}) needs "
                f"a {want} build (wrong backend wheel)", fix, severity=2))
        elif gpu.driver_ready and not (venv.torch_has_cuda or venv.torch_xpu or venv.torch_mps):
            out.append(Issue(
                REINSTALL_TORCH,
                f"torch {venv.torch_version} is a {kind} build but cannot see the GPU "
                f"(torch.cuda/xpu.is_available() == False; driver {gpu.driver_version or 'n/a'})",
                fix + "; if it still fails the driver is too old/new for this wheel", severity=2))
        elif kind == "cuda" and plan.torch_tag and "+" + plan.torch_tag not in venv.torch_version \
                and gpu.vendor == "nvidia" and "+cu" not in venv.torch_version:
            out.append(Issue(REINSTALL_TORCH,
                             f"torch {venv.torch_version} has CUDA but not the {plan.torch_tag} tag", fix, severity=1))
    elif gpu.vendor == "none" and not force_cpu and kind != "cpu" and "+cpu" not in venv.torch_version:
        out.append(Issue(
            REINSTALL_TORCH, f"torch {venv.torch_version} has a GPU build but no GPU detected",
            "reinstall torch CPU build:  uv pip install --reinstall --index-url "
            "https://download.pytorch.org/whl/cpu torch", severity=1))

    # torchaudio (mixed-install canary; only when installed)
    if venv.torchaudio_importable is False:
        out.append(Issue(
            REINSTALL_TORCH,
            "torchaudio cannot import (mixed torch install — libc10_cuda.so missing). "
            "Almost always means torch is +cpu but torchaudio is +cuXXX.",
            "reinstall the torch family in one shot:  " + fix.replace("reinstall torch:  ", ""), severity=2))
    return out


def _check_llama_cpp_python(venv: VenvInfo, gpu: GpuInfo, plan: ap.Plan) -> list[Issue]:
    out: list[Issue] = []
    if not venv.llama_cpp_version:
        out.append(Issue(REINSTALL_LLAMA_CPP, "llama-cpp-python not installed (GGUF inference unavailable)",
                         "install:  bash install.sh   (or:  uv pip install 'llama-cpp-python>=0.3.0')", severity=2))
        return out
    best, _ = ap.best_llama_backend(plan)
    if gpu.vendor == "none":
        return out
    offload = probe_llama_offload(venv.python) if venv.python else None
    if offload is False:
        why = (f"expected {best.name} offload" if best.name != "cpu" else
               "no GPU backend toolchain on this host (see the toolchain issue)")
        out.append(Issue(
            REINSTALL_LLAMA_CPP,
            f"llama-cpp-python {venv.llama_cpp_version} has NO GPU offload (CPU build) but a "
            f"{gpu.vendor.upper()} GPU is present — {why}.",
            "reinstall with the right backend:  bash install.sh --repair   (builds with "
            + (" ".join(best.cmake_args) or "a GPU backend") + ")", severity=2))
    elif gpu.vendor == "nvidia" and gpu.compute_cap:
        # Blackwell sm_100/120: abetlen prebuilt wheels lack those kernels (crash in
        # ggml_cuda_op_scale on the first token) -> need OUR source build for this arch.
        try:
            cc_major = int(float(gpu.compute_cap.split(".")[0]))
        except ValueError:
            cc_major = 0
        marker = _llama_build_marker(venv.path)
        if cc_major >= 10 and offload is not False and not (
                marker.get("source") and f"{cc_major}0" in str(marker.get("arch", ""))):
            out.append(Issue(
                REINSTALL_LLAMA_CPP,
                f"GPU {gpu.name} (sm_{cc_major}0/Blackwell) needs a llama-cpp-python source build "
                f"with sm_{cc_major}0 kernels — prebuilt wheels crash on the first token.",
                f"rebuild from source:  CMAKE_ARGS=\"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={cc_major}0\" "
                f"uv pip install --reinstall-package llama-cpp-python --no-binary llama-cpp-python llama-cpp-python",
                severity=2))
    return out


def _check_llama_cpp_cli(llcpp: dict, gpu: GpuInfo, plan: ap.Plan) -> list[Issue]:
    out: list[Issue] = []
    if not (llcpp["quantize_exists"] and llcpp["convert_exists"]):
        out.append(Issue(
            BUILD_LLAMA_CPP_CLI,
            f"llama.cpp CLI missing at {llcpp['dir']} "
            f"(quantize={'yes' if llcpp['quantize_exists'] else 'NO'}, "
            f"convert={'yes' if llcpp['convert_exists'] else 'NO'}). Needed for GGUF export endpoint.",
            "build:  bash install.sh   (or:  bash install.sh --llama-cpp-only)", severity=2))
        return out
    best, _ = ap.best_llama_backend(plan)
    if llcpp.get("backend") == "cpu" and best.name != "cpu":
        out.append(Issue(
            BUILD_LLAMA_CPP_CLI,
            f"llama.cpp CLI at {llcpp['dir']} was built CPU-only but this host can build the "
            f"{best.name} backend ({gpu.name}).",
            "rebuild:  bash install.sh --llama-cpp-only --rebuild-llama-cpp", severity=2))
    elif not llcpp.get("cli_exists", True):
        out.append(Issue(BUILD_LLAMA_CPP_CLI, f"llama-cli binary missing in {llcpp['dir']}/build/bin",
                         "rebuild:  bash install.sh --llama-cpp-only --rebuild-llama-cpp", severity=1))
    return out


def _check_gpu_prereqs(venv: VenvInfo, gpu: GpuInfo, plan: ap.Plan) -> list[Issue]:
    """Driver / toolchain / bitsandbytes: things around the GPU stack (some need root)."""
    out: list[Issue] = []
    if gpu.vendor in ("nvidia", "amd", "intel") and not gpu.driver_ready:
        out.append(Issue(INSTALL_GPU_DRIVER, gpu.hint or f"{gpu.vendor} GPU without a working driver",
                         "install the vendor driver (needs sudo + reboot), then:  bash install.sh --repair",
                         severity=2))
    for b in plan.llama_backends:  # report every preferred backend that is unbuildable, up to the first that is
        if not b.missing:
            break
        if b.name != "cpu":
            out.append(Issue(INSTALL_GPU_TOOLCHAIN, f"llama.cpp {b.name} backend cannot be built: {b.missing}",
                             b.missing, severity=1))
    if gpu.vendor in ("nvidia",) and not gpu.cuda_toolkit_path and not gpu.nvcc:
        out.append(Issue(
            INSTALL_CUDA_TOOLKIT,
            "NVIDIA GPU detected but no CUDA toolkit in /opt/cuda, /usr/local/cuda, or /usr/lib/cuda. "
            "Source builds (llama.cpp CUDA, Blackwell wheels) will fail.",
            "install the CUDA toolkit matching your driver and ensure /usr/local/cuda exists.", severity=1))
    if plan.bnb and venv.python and probe_bitsandbytes(venv.python) is None:
        out.append(Issue(INSTALL_BITSANDBYTES, "bitsandbytes not installed (4-bit/8-bit model loading unavailable)",
                         "install:  bash install.sh --repair   (or:  uv pip install bitsandbytes)", severity=1))
    return out


# ── Main diagnose() ─────────────────────────────────────────────────────

def diagnose(
    venv_dir: Path,
    llama_cpp_dir: Path,
    *,
    force_cpu: bool = False,
    check_service: bool = True,
) -> list[Issue]:
    """Return a list of issues found. Empty list = fully healthy."""
    issues: list[Issue] = []
    gpu = GpuInfo.detect(force_cpu=force_cpu)
    venv = inspect_venv(venv_dir)
    llcpp = inspect_llama_cpp(llama_cpp_dir)

    # ── Venv-level checks ──
    if not venv.python:
        issues.append(Issue(
            RECREATE_VENV,
            f"venv python missing at {venv.path}/bin/python",
            "recreate venv:  bash install.sh   (or   bash update.sh --repair)",
            severity=3,
        ))
        return issues  # nothing else can be checked without a venv python

    if venv.py_version is None:
        issues.append(Issue(
            RECREATE_VENV,
            "venv python cannot execute even basic sys/os imports",
            f"recreate venv at {venv.path}",
            severity=3,
        ))
        return issues

    # ── Key deps ──
    key_deps = (
        ("fastapi", venv.has_fastapi), ("jinja2", venv.has_jinja2),
        ("datasets", venv.has_datasets), ("accelerate", venv.has_accelerate),
        ("safetensors", venv.has_safetensors),
        ("huggingface_hub", venv.has_huggingface_hub),
    )
    if not all(has for _, has in key_deps):
        missing = [n for n, has in key_deps if not has]
        issues.append(Issue(
            PIP_INSTALL_EDITABLE,
            f"missing or broken: {', '.join(missing)}",
            "sync deps:  bash update.sh   (or:  pip install -e .)",
            severity=2,
        ))

    if venv.missing_parsers:
        lost = ", ".join(f"{m} ({_PARSER_MODULES[m]})" for m in venv.missing_parsers)
        issues.append(Issue(
            PIP_INSTALL_EDITABLE,
            f"document parsers missing: {lost}",
            "sync deps:  bash install.sh --repair   (or:  uv pip install -c .venv/torch-constraints.txt -e '.[parsers]')",
            severity=1,
        ))

    plan = ap.build_plan(gpu)
    issues += _check_torch(venv, gpu, plan, force_cpu)
    issues += _check_llama_cpp_python(venv, gpu, plan)
    issues += _check_llama_cpp_cli(llcpp, gpu, plan)
    issues += _check_gpu_prereqs(venv, gpu, plan)

    # ── systemd service (optional but recommended for fan-dragon / servers) ──
    if check_service:
        svc = inspect_service()
        if svc["systemd_available"] and not svc["user_unit_installed"]:
            # Only flag this when running from a repo checkout (install.sh
            # path detection). Auto-install is opt-in via install-service.sh.
            install_svc = Path(__file__).resolve().parent.parent / "install-service.sh"
            if install_svc.exists():
                issues.append(Issue(
                    INSTALL_SERVICE,
                    "finetune-studio systemd user-unit is not installed. "
                    "The webui is currently running as a bare nohup/uvicorn "
                    "process; it won't auto-restart on crash or reboot.",
                    "install:  bash install-service.sh",
                    severity=1,
                ))

    return issues


def _run_step(
    label: str,
    cmd: list[str],
    timeout: int,
    actions: list[str],
    log_fn: Callable[[str], None],
    *,
    env: dict[str, str] | None = None,
    tail: int = 800,
) -> bool:
    """Run one repair command; record ``"<label>: ok|FAIL"`` and log failure output.

    A command that cannot run at all (spawn error, timeout) is a recorded
    failure, never an exception that aborts the remaining repairs."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
            **({"env": env} if env is not None else {}),
        )
    except _STEP_ERRORS as exc:
        actions.append(f"{label}: FAIL ({type(exc).__name__}: {exc})")
        log_fn(f"[repair] {label} could not run: {exc}")
        return False
    ok = r.returncode == 0
    actions.append(f"{label}: {'ok' if ok else 'FAIL'}")
    if not ok:
        log_fn(((r.stderr or r.stdout) or "")[-tail:])
    return ok


# ── Repair ──────────────────────────────────────────────────────────────

def repair(
    issues: Iterable[Issue],
    venv_dir: Path,
    llama_cpp_dir: Path,
    *,
    log: Callable[[str], None] = print,
    force: bool = False,
) -> tuple[bool, list[str]]:
    """Apply fixes for each issue in dependency order.

    Returns (ok, [summary_lines]). ok=True if all fixes succeeded or
    no fixes were needed. ok=False if any fix failed.

    `force=True` will recreate venv even for non-critical issues
    (used by --repair mode). Otherwise RECREATE_VENV is only applied
    if severity >= 3."""
    actions: list[str] = []
    issues = list(issues)
    gpu = GpuInfo.detect()
    venv_py = venv_dir / "bin" / "python"

    if any(i.code == RECREATE_VENV for i in issues):
        log(f"[repair] recreating venv at {venv_dir}")
        shutil.rmtree(venv_dir, ignore_errors=True)
        actions.append("removed venv")
        # Delegate full recreate to install.sh — it knows the GPU-aware
        # wheel index for this host. Caller is expected to re-run
        # install.sh after this function returns.
        actions.append("DEFER: run 'bash install.sh' to recreate venv")
        return False, actions

    # Order: torch family → llama-cpp → pyproject deps → llama.cpp CLI.
    # `all_ok` is sticky: a later successful step can never hide an earlier failure.
    all_ok = True
    by_code: dict[str, list[Issue]] = {}
    for i in issues:
        by_code.setdefault(i.code, []).append(i)

    def step(label: str, cmd: list[str], timeout: int, *, env: dict[str, str] | None = None,
             tail: int = 800) -> bool:
        nonlocal all_ok
        ok = _run_step(label, cmd, timeout, actions, log, env=env, tail=tail)
        all_ok = all_ok and ok
        return ok

    plan = ap.build_plan(gpu)
    constraints = ap.constraint_args(venv_dir)
    repo_root = Path(__file__).resolve().parent.parent
    py = str(venv_py)

    def chain(label: str, attempts: list[tuple[str, list[str], dict[str, str], list[str] | None]],
              timeout: int, *, tail: int = 800) -> str | None:
        """Try each (sub-label, cmd, env, verify-cmd) until one installs AND verifies; returns
        the winning sub-label (None = all failed). One recorded line per chain; a later success
        never hides an earlier chain's failure."""
        nonlocal all_ok
        scratch: list[str] = []
        for sub, cmd, env, verify in attempts:
            run_env = {**os.environ, **env} if env else None
            if not _run_step(f"{label} [{sub}]", cmd, timeout, scratch, log, env=run_env, tail=tail):
                continue
            if verify and not _run_step(f"{label} verify [{sub}]", verify, 180, scratch, log, tail=tail):
                log(f"[repair] {label}: {sub} installed but did not verify -- trying the next option")
                continue
            actions.append(f"{label}: ok" + (f" ({sub})" if len(attempts) > 1 else ""))
            return sub
        actions.append(f"{label}: FAIL")
        all_ok = False
        return None

    if REINSTALL_TORCH in by_code:
        label = "torch reinstall" if plan.vendor != "none" else "torch CPU reinstall"
        log(f"[repair] reinstalling torch ({plan.torch_tag or 'default'}, {plan.torch_backend}) -- "
            f"CPU wheels only when no GPU exists (vendor={plan.vendor})")
        verify = [py, "-c", ap.torch_probe_code(plan.torch_backend)]
        if not chain(label, [(t or "pypi", c, {}, verify) for t, c in ap.torch_commands(plan, py)], 900):
            if plan.vendor != "none":
                return False, actions  # a GPU host without its GPU torch: later steps would only mask it
        else:
            ap.write_constraints(venv_dir)  # re-pin to the freshly installed family
            constraints = ap.constraint_args(venv_dir)

    if REINSTALL_LLAMA_CPP in by_code:
        log(f"[repair] reinstalling llama-cpp-python (GPU vendor={plan.vendor}: "
            f"{' > '.join(b.name for b in plan.llama_backends if not b.missing)})")
        attempts = []
        for sub, cmd, env in ap.llama_py_attempts(plan, py, constraints):
            gpu_try = plan.vendor != "none" and sub != "CPU build"
            attempts.append((sub, cmd, env, [py, "-c", ap.LLAMA_PROBE] if gpu_try else None))
        won = chain("llama-cpp reinstall", attempts, 1800, tail=1200)
        if won:
            ap.write_llama_marker(venv_dir, plan, won)

    if PIP_INSTALL_EDITABLE in by_code:
        log("[repair] syncing editable deps (uv pip install -e .[parsers])")
        step("pip -e .", ap.uv_pip(py, *constraints, "-e", f"{repo_root}[parsers]"), 600)

    if INSTALL_BITSANDBYTES in by_code:
        log("[repair] installing bitsandbytes")
        step("bitsandbytes", ap.uv_pip(py, *constraints, "bitsandbytes>=0.45.5"), 600)

    if BUILD_LLAMA_CPP_CLI in by_code:
        # Delegate to bash — install.sh owns the cmake + pip-deps logic.
        # --llama-cpp-only skips torch / pyproject work and just builds;
        # --rebuild-llama-cpp replaces a CPU-only / wrong-backend build.
        log("[repair] building llama.cpp CLI (cmake + clone if missing)")
        install_sh = Path(__file__).parent.parent / "install.sh"
        step("llama.cpp CLI build",
             ["bash", str(install_sh), "--llama-cpp-only", "--rebuild-llama-cpp"], 1800, tail=1200)

    for code in MANUAL_CODES | {INSTALL_CUDA_TOOLKIT}:
        for i in by_code.get(code, []):
            # Needs root (driver / toolkit packages): never run sudo from an installer.
            actions.append(f"{code}: MANUAL -- {i.suggested_fix}")
            log(f"[repair] {code}: {i.detail}\n         -> {i.suggested_fix}")

    if INSTALL_SERVICE in by_code:
        # Delegate to install-service.sh which already handles --user/--system,
        # enable-linger, etc.
        log("[repair] installing finetune-studio systemd service")
        service_sh = Path(__file__).parent.parent / "install-service.sh"
        step("systemd service", ["bash", str(service_sh)], 300)

    return all_ok, actions


# ── CLI (so bash can call this without imports) ────────────────────────

def _main(argv: list[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Finetune Studio install diagnostics.")
    parser.add_argument("--venv", default=".venv", help="venv directory (default: .venv)")
    parser.add_argument("--llama-cpp", default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".llama.cpp"),
                        help="llama.cpp checkout directory (project-local default)")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--check", action="store_true",
                        help="exit 0 if healthy, 1 if warnings, 2+ if broken")
    parser.add_argument("--repair", action="store_true",
                        help="apply fixes for detected issues")
    parser.add_argument("--force-recreate", action="store_true",
                        help="with --repair, nuke and recreate venv")
    parser.add_argument("--no-service-check", action="store_true",
                        help="skip systemd unit check")
    parser.add_argument("--cpu", action="store_true", help="force CPU mode")
    parser.add_argument("--gpu", default="", choices=["", "nvidia", "amd", "intel", "apple", "none"],
                        help="override accelerator detection")
    args = parser.parse_args(argv)

    venv = Path(args.venv).resolve()
    llcpp = Path(args.llama_cpp).resolve()
    if args.gpu:
        os.environ["FTS_FORCE_VENDOR"] = args.gpu  # read by accel_plan.detect()
    issues = diagnose(venv, llcpp, force_cpu=args.cpu,
                      check_service=not args.no_service_check)

    if args.json:
        print(json.dumps([i.to_dict() for i in issues], indent=2))
    else:
        _print_report(venv, llcpp, issues, args.cpu)

    if args.repair:
        ok, actions = repair(issues, venv, llcpp, force=args.force_recreate)
        if args.json:
            print(json.dumps({"ok": ok, "actions": actions}))
        else:
            for a in actions:
                print(f"[repair] {a}")
        # After repair, re-diagnose
        issues = diagnose(venv, llcpp, force_cpu=args.cpu,
                          check_service=not args.no_service_check)
        critical = any(i.severity >= 3 for i in issues)
        return 1 if critical or not ok else 0

    critical = any(i.severity >= 2 for i in issues)
    warn = any(i.severity == 1 for i in issues)
    # --check and the default print mode share semantics: 2 = broken, 1 = warnings,
    # so `python install_diagnose.py && echo healthy` works in shell scripts.
    return 2 if critical else (1 if warn else 0)


def _print_report(venv: Path, llcpp: Path, issues: list[Issue], force_cpu: bool) -> None:
    gpu = GpuInfo.detect(force_cpu=force_cpu)
    plan = ap.build_plan(gpu)
    best, _ = ap.best_llama_backend(plan)
    venv_info = inspect_venv(venv)
    llcpp_info = inspect_llama_cpp(llcpp)
    ta = ("n/a" if venv_info.torchaudio_importable is None else
          "ok" if venv_info.torchaudio_importable else "BROKEN")
    print(f"venv:        {venv}  ({'OK' if venv_info.healthy else 'BROKEN'})")
    print(f"  py={venv_info.py_version}  torch={venv_info.torch_version} ({venv_info.torch_kind or '?'})  "
          f"gpu_visible={venv_info.torch_has_cuda or venv_info.torch_xpu or venv_info.torch_mps}  "
          f"torchaudio={ta} ({venv_info.torchaudio_version or '-'})")
    print(f"  fastapi={'ok' if venv_info.has_fastapi else 'MISSING'}  "
          f"llama_cpp={venv_info.llama_cpp_version or 'MISSING'}  "
          f"peft={venv_info.peft_version or 'MISSING'}  "
          f"trl={venv_info.trl_version or 'MISSING'}  "
          f"transformers={venv_info.transformers_version or 'MISSING'}")
    print(f"GPU:         {gpu.vendor}: {gpu.name}  driver={gpu.driver_version or 'n/a'}  "
          f"cuda_max={gpu.cuda_max or 'n/a'}  cc={gpu.compute_cap or 'n/a'}  "
          f"toolkit={'yes' if gpu.cuda_toolkit_path else 'NO'}")
    print(f"plan:        torch={plan.torch_tag or 'default'} ({plan.torch_backend})  "
          f"llama.cpp backend={best.name}")
    print(f"llama.cpp:   dir={llcpp}  "
          f"quantize={'yes' if llcpp_info['quantize_exists'] else 'NO'}  "
          f"convert={'yes' if llcpp_info['convert_exists'] else 'NO'}  "
          f"backend={llcpp_info['backend'] or '?'}")
    if not issues:
        print("issues:      none ✓")
        return
    print(f"issues:      {len(issues)}")
    for i in issues:
        print(f"  [{i.severity}] {i.code}: {i.detail}")
        print(f"        fix: {i.suggested_fix}")


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
