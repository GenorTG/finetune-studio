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
  - llama-cpp-python not importable                   → REINSTALL_LLAMA_CPP
  - llama.cpp CLI tools missing (llama-quantize)      → BUILD_LLAMA_CPP_CLI
  - pyproject deps missing                           → PIP_INSTALL_EDITABLE
  - CUDA toolkit missing on GPU host                  → INSTALL_CUDA_TOOLKIT
  - systemd unit missing on a host that should run it → INSTALL_SERVICE
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


# ── GPU detection (mirrors install.sh but in Python) ────────────────────

@dataclass(frozen=True)
class GpuInfo:
    vendor: str            # "nvidia" | "amd" | "intel" | "none"
    name: str
    driver_version: str
    cuda_ver: str          # "cu130" / "cu124" / "cu121" / "cu118" / ""
    compute_cap: str       # "12.0" for sm_120 (Blackwell), "8.6" for sm_86, etc.
    cuda_toolkit_path: str # "/opt/cuda" if found

    @classmethod
    def detect(cls, force_cpu: bool = False) -> GpuInfo:
        vendor = "none"
        name = "(no GPU)"
        driver = ""
        cuda_ver = ""
        cuda_path = ""
        compute_cap = ""

        # CUDA toolkit path (used by llama-cpp source builds + sanity checks)
        for cand in ("/opt/cuda", "/usr/local/cuda", "/usr/lib/cuda"):
            if Path(cand).exists():
                cuda_path = cand
                break

        if force_cpu:
            return cls("none", "(forced CPU)", "", "", "", cuda_path)

        # NVIDIA
        if shutil.which("nvidia-smi"):
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,driver_version",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10, check=False,
                )
                if r.returncode == 0 and r.stdout.strip():
                    line = r.stdout.strip().splitlines()[0]
                    parts = [p.strip() for p in line.split(",", 1)]
                    name = parts[0] if parts else "NVIDIA GPU"
                    driver = parts[1] if len(parts) > 1 else ""
                    vendor = "nvidia"
                    # Driver → CUDA mapping (mirrors install.sh)
                    try:
                        major = int(driver.split(".")[0]) if driver else 0
                    except ValueError:
                        major = 0
                    if   major >= 550: cuda_ver = "cu130"
                    elif major >= 525: cuda_ver = "cu124"
                    elif major >= 520: cuda_ver = "cu121"
                    elif major >= 470: cuda_ver = "cu118"
                    else:               cuda_ver = "cu118"
                # Compute capability (e.g. "12.0" for sm_120/Blackwell,
                # "8.6" for sm_86/Ampere). Prebuilt llama-cpp-python wheels
                # only bundle the kernels the maintainer chose to compile;
                # sm_120 needs a source build. Detected here so diagnose()
                # can emit REINSTALL_LLAMA_CPP for unsupported arches.
                r2 = subprocess.run(
                    ["nvidia-smi", "--query-gpu=compute_cap",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10, check=False,
                )
                if r2.returncode == 0 and r2.stdout.strip():
                    compute_cap = r2.stdout.strip().splitlines()[0].strip()
                else:
                    compute_cap = ""
            except (subprocess.TimeoutExpired, OSError) as exc:
                log.debug("nvidia-smi probe failed: %s", exc)

        # AMD ROCm
        if vendor == "none" and (shutil.which("rocm-smi") or Path("/opt/rocm").exists()):
            vendor = "amd"
            name = "AMD GPU (ROCm)"
            compute_cap = ""

        # Intel XPU
        if vendor == "none" and (
            shutil.which("xpu-smi") or Path("/opt/intel/oneapi").exists()
        ):
            vendor = "intel"
            name = "Intel XPU"
            compute_cap = ""

        return cls(vendor, name, driver, cuda_ver, compute_cap, cuda_path)


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
            and (self.torchaudio_importable is True)
            and self.torch_has_cuda in (True, None)  # None = no GPU expected
        )


def _run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False,
        env={**os.environ, "PYTHONPATH": "", "VIRTUAL_ENV": ""},
    )


_DEP_MODULES = ("fastapi", "jinja2", "datasets", "accelerate", "safetensors", "huggingface_hub")
_DEPS_PROBE = (
    "import importlib, json, sys\n"
    f"mods = {list(_DEP_MODULES)!r}\n"
    "missing = []\n"
    "for m in mods:\n"
    "    try:\n"
    "        importlib.import_module(m)\n"
    "    except Exception:\n"
    "        missing.append(m)\n"
    "print(json.dumps({'missing': missing}))\n"
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
        r = _run([str(info.python), "-c",
                  ("import torch, json; print(json.dumps("
                  "{'v': torch.__version__, 'cuda': torch.cuda.is_available()}"
                  "))")], timeout=45)
        if r.returncode == 0 and r.stdout.strip():
            j = json.loads(r.stdout.strip().splitlines()[-1])
            info.torch_version = j["v"]
            info.torch_has_cuda = bool(j["cuda"])
    except _PROBE_ERRORS as exc:
        log.debug("torch probe failed: %s", exc)

    # torchaudio — separate so a broken torchaudio doesn't mask torch status
    try:
        r = _run([str(info.python), "-c",
                  "import torchaudio; print(torchaudio.__version__)"],
                  timeout=30)
        if r.returncode == 0:
            info.torchaudio_version = r.stdout.strip().splitlines()[-1]
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
    convert = llama_cpp_dir / "convert_hf_to_gguf.py"
    return {
        "dir": str(llama_cpp_dir),
        "quantize_exists": quantize.exists() and os.access(quantize, os.X_OK),
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

    # ── Torch + CUDA ──
    if venv.torch_version is None:
        issues.append(Issue(
            REINSTALL_TORCH,
            "torch not importable",
            f"reinstall torch with matching CUDA index (gpu={gpu.vendor}, cuda={gpu.cuda_ver or 'n/a'})",
            severity=3,
        ))
    else:
        if gpu.vendor == "nvidia" and gpu.cuda_ver:
            expected = gpu.cuda_ver  # e.g. "cu130"
            if "+cpu" in venv.torch_version:
                issues.append(Issue(
                    REINSTALL_TORCH,
                    f"torch {venv.torch_version} is CPU-only but NVIDIA GPU "
                    f"({gpu.name}, driver {gpu.driver_version}) needs {expected}",
                    f"reinstall torch:  uv pip install --python {venv.python} "
                    f"--reinstall --index-url https://download.pytorch.org/whl/{expected} "
                    f"torch torchvision torchaudio",
                    severity=2,
                ))
            elif not venv.torch_has_cuda:
                issues.append(Issue(
                    REINSTALL_TORCH,
                    f"torch {venv.torch_version} is installed but "
                    f"torch.cuda.is_available() == False (driver "
                    f"{gpu.driver_version} / cuda {expected})",
                    f"reinstall torch for {expected}; if driver < 470, upgrade "
                    f"NVIDIA driver first",
                    severity=2,
                ))
            elif "+" + expected not in venv.torch_version and \
                 "+cu" not in venv.torch_version:
                # Has CUDA but mismatched cu-tag — sometimes OK, sometimes not
                issues.append(Issue(
                    REINSTALL_TORCH,
                    f"torch {venv.torch_version} has CUDA but the cu-tag "
                    f"doesn't match driver ({expected}). May work, but "
                    f"a matching build is safer.",
                    f"reinstall torch for {expected} (or run "
                    f"`bash install.sh --check` to confirm)",
                    severity=1,
                ))
        elif gpu.vendor == "none" and not force_cpu:
            if "+cpu" not in venv.torch_version:
                issues.append(Issue(
                    REINSTALL_TORCH,
                    f"torch {venv.torch_version} has GPU build but no GPU detected",
                    "reinstall torch CPU build:  uv pip install --reinstall "
                    "--index-url https://download.pytorch.org/whl/cpu torch",
                    severity=1,
                ))

    # ── torchaudio (mixed-install canary) ──
    if venv.torch_version and venv.torchaudio_importable is False:
        issues.append(Issue(
            REINSTALL_TORCH,
            "torchaudio cannot import (mixed torch install — "
            "libc10_cuda.so missing). Almost always means torch is "
            "+cpu but torchaudio is +cuXXX.",
            f"reinstall the whole torch family in one shot:  uv pip install "
            f"--python {venv.python} --reinstall "
            f"--index-url https://download.pytorch.org/whl/"
            f"{gpu.cuda_ver or 'cpu'} torch torchvision torchaudio",
            severity=2,
        ))

    # ── llama-cpp-python ──
    if not venv.llama_cpp_version:
        issues.append(Issue(
            REINSTALL_LLAMA_CPP,
            "llama-cpp-python not installed (GGUF inference unavailable)",
            "install:  bash install.sh   (or:  uv pip install 'llama-cpp-python>=0.3.0')",
            severity=2,
        ))
    elif gpu.vendor == "nvidia" and gpu.cuda_ver:
        # Heuristic: a CPU llama-cpp wheel is usually < 30 MiB. A CUDA
        # wheel is usually > 100 MiB. We can't be 100% sure without
        # importing, so we just log a warning if the version is suspiciously
        # old and a GPU is present.
        try:
            v = tuple(int(x) for x in venv.llama_cpp_version.split(".")[:2])
            if v < (0, 3):
                issues.append(Issue(
                    REINSTALL_LLAMA_CPP,
                    f"llama-cpp-python {venv.llama_cpp_version} is old; GGUF "
                    f"support is unreliable on this version with newer cu tags.",
                    "upgrade:  uv pip install --reinstall 'llama-cpp-python>=0.3.0'",
                    severity=1,
                ))
        except ValueError:
            pass

    # ── Blackwell sm_120 source-build requirement ──
    # Prebuilt llama-cpp-python wheels from abetlen don't bundle sm_120
    # kernels (verified: crash in ggml_cuda_op_scale on RTX 5090/5080
    # at the first forward pass). The wheel install path silently produces
    # a "loaded" model that ABRTs on the first chat token. Source build
    # with CMAKE_CUDA_ARCHITECTURES=120 fixes it.
    if gpu.vendor == "nvidia" and gpu.compute_cap:
        try:
            cc_major = int(float(gpu.compute_cap.split(".")[0]))
        except ValueError:
            cc_major = 0
        if cc_major >= 12 and venv.llama_cpp_version:
            issues.append(Issue(
                REINSTALL_LLAMA_CPP,
                f"GPU {gpu.name} (sm_{cc_major}0/Blackwell) requires source-"
                f"rebuilt llama-cpp-python — prebuilt abetlen wheels lack "
                f"sm_{cc_major}0 CUDA kernels (crashes in ggml_cuda_op_scale "
                f"on first token).",
                f"rebuild from source:  CMAKE_ARGS=\"-DGGML_CUDA=ON "
                f"-DCMAKE_CUDA_ARCHITECTURES={cc_major}0\" "
                f"pip install --force-reinstall --no-deps llama-cpp-python",
                severity=2,
            ))

    # ── llama.cpp CLI ──
    if not (llcpp["quantize_exists"] and llcpp["convert_exists"]):
        issues.append(Issue(
            BUILD_LLAMA_CPP_CLI,
            f"llama.cpp CLI missing at {llcpp['dir']} "
            f"(quantize={'yes' if llcpp['quantize_exists'] else 'NO'}, "
            f"convert={'yes' if llcpp['convert_exists'] else 'NO'}). "
            f"Needed for GGUF export endpoint.",
            "build:  bash install.sh   (or:  bash install.sh --llama-cpp-only)",
            severity=2,
        ))

    # ── CUDA toolkit (only relevant for source builds of llama-cpp / torch) ──
    if gpu.vendor == "nvidia" and not gpu.cuda_toolkit_path:
        issues.append(Issue(
            INSTALL_CUDA_TOOLKIT,
            "NVIDIA GPU detected but no CUDA toolkit in /opt/cuda, "
            "/usr/local/cuda, or /usr/lib/cuda. Source builds will fail.",
            "install CUDA toolkit (matches driver) and ensure /opt/cuda exists.",
            severity=1,
        ))

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

    if REINSTALL_TORCH in by_code:
        if gpu.vendor == "nvidia" and gpu.cuda_ver:
            idx = f"https://download.pytorch.org/whl/{gpu.cuda_ver}"
            log(f"[repair] reinstalling torch family from {idx}")
            if not step("torch reinstall", [
                str(venv_py), "-m", "pip", "install",
                "--reinstall", "--index-url", idx,
                "torch", "torchvision", "torchaudio",
            ], 900):
                return False, actions
        else:
            idx = "https://download.pytorch.org/whl/cpu"
            log(f"[repair] reinstalling torch (CPU) from {idx}")
            step("torch CPU reinstall", [
                str(venv_py), "-m", "pip", "install", "--reinstall",
                "--index-url", idx, "torch", "torchvision", "torchaudio",
            ], 900)

    if REINSTALL_LLAMA_CPP in by_code:
        # Blackwell sm_120 needs source build (prebuilt wheels lack kernels);
        # other NVIDIA cards use abetlen's prebuilt wheel for speed.
        sm_blackwell = False
        cc_major = 0
        if gpu.vendor == "nvidia" and gpu.compute_cap:
            try:
                cc_major = int(float(gpu.compute_cap.split(".")[0]))
            except ValueError:
                cc_major = 0
            sm_blackwell = cc_major >= 12

        if sm_blackwell:
            arch = f"{cc_major}0"
            log(f"[repair] rebuilding llama-cpp-python from source for sm_{arch} (Blackwell)")
            env = os.environ.copy()
            env["CMAKE_ARGS"] = f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={arch}"
            # --no-binary llama-cpp-python forces source build even when
            # a matching wheel is in pip's cache (the cached abetlen wheel
            # is the broken one — no sm_120 kernels). --no-deps prevents
            # torch/peft/etc from being touched.
            step(f"llama-cpp source build (sm_{arch})", [
                str(venv_py), "-m", "pip", "install",
                "--force-reinstall", "--no-deps",
                "--no-binary", "llama-cpp-python",
                "llama-cpp-python",
            ], 1800, env=env, tail=1200)
        elif gpu.vendor == "nvidia" and gpu.cuda_ver:
            log(f"[repair] reinstalling llama-cpp-python (cuda {gpu.cuda_ver})")
            step("llama-cpp reinstall", [
                str(venv_py), "-m", "pip", "install", "--reinstall",
                "--extra-index-url",
                (f"https://abetlen.github.io/llama-cpp-python/whl/"
                f"{gpu.cuda_ver}/llama-cpp-python/"),
                "llama-cpp-python>=0.3.0",
            ], 900)
        else:
            log("[repair] reinstalling llama-cpp-python (CPU)")
            step("llama-cpp reinstall", [
                str(venv_py), "-m", "pip", "install", "--reinstall",
                "llama-cpp-python>=0.3.0",
            ], 900)

    if PIP_INSTALL_EDITABLE in by_code:
        log("[repair] syncing editable deps (pip install -e .)")
        step("pip -e .", [str(venv_py), "-m", "pip", "install", "-e", "."], 600)

    if BUILD_LLAMA_CPP_CLI in by_code:
        # Delegate to bash — install.sh owns the cmake + pip-deps logic.
        # --llama-cpp-only skips torch / pyproject work and just builds.
        log("[repair] building llama.cpp CLI (cmake + clone if missing)")
        install_sh = Path(__file__).parent.parent / "install.sh"
        step("llama.cpp CLI build", ["bash", str(install_sh), "--llama-cpp-only"], 1800, tail=1200)

    if INSTALL_CUDA_TOOLKIT in by_code:
        # Detect distro + run the right package manager. The toolkit is
        # required by the cu118/cu124/cu130 PyTorch wheels at runtime,
        # even though we only need the driver for nvidia-smi to work.
        log("[repair] CUDA toolkit missing on an NVIDIA host — installing")
        distro = _detect_distro()
        cmd_map = {
            "debian": ["sudo", "apt-get", "install", "-y", "cuda-toolkit-12-4"],
            "ubuntu": ["sudo", "apt-get", "install", "-y", "cuda-toolkit-12-4"],
            "fedora": ["sudo", "dnf", "install", "-y", "cuda-toolkit-12-4"],
            "arch":   ["sudo", "pacman", "-S", "--noconfirm", "cuda"],
            "garuda": ["sudo", "pacman", "-S", "--noconfirm", "cuda"],
        }
        cmd = cmd_map.get(distro)
        if cmd is None:
            actions.append(
                f"cuda toolkit: SKIP (distro '{distro}' not handled; install manually)"
            )
        else:
            step(f"cuda toolkit ({distro})", cmd, 900, tail=1200)

    if INSTALL_SERVICE in by_code:
        # Delegate to install-service.sh which already handles --user/--system,
        # enable-linger, etc.
        log("[repair] installing finetune-studio systemd service")
        service_sh = Path(__file__).parent.parent / "install-service.sh"
        step("systemd service", ["bash", str(service_sh)], 300)

    return all_ok, actions


def _detect_distro() -> str:
    """Best-effort distro detection. Returns one of: debian, ubuntu,
    fedora, arch, garuda, macos, unknown."""
    if Path("/etc/os-release").exists():
        try:
            with open("/etc/os-release") as f:
                os_release = {}
                for line in f:
                    if "=" in line:
                        k, v = line.split("=", 1)
                        os_release[k.strip()] = v.strip().strip('"').lower()
            id_ = os_release.get("id", "")
            return id_ if id_ in {"debian", "ubuntu", "fedora", "arch", "garuda"} else "unknown"
        except OSError as exc:
            log.debug("cannot read /etc/os-release: %s", exc)
    return "unknown"


# ── CLI (so bash can call this without imports) ────────────────────────

def _main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Finetune Studio install diagnostics.")
    ap.add_argument("--venv", default=".venv", help="venv directory (default: .venv)")
    ap.add_argument("--llama-cpp", default=os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".llama.cpp"),
                    help="llama.cpp checkout directory (project-local default)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--check", action="store_true",
                    help="exit 0 if healthy, 1 if warnings, 2+ if broken")
    ap.add_argument("--repair", action="store_true",
                    help="apply fixes for detected issues")
    ap.add_argument("--force-recreate", action="store_true",
                    help="with --repair, nuke and recreate venv")
    ap.add_argument("--no-service-check", action="store_true",
                    help="skip systemd unit check")
    ap.add_argument("--cpu", action="store_true", help="force CPU mode")
    args = ap.parse_args(argv)

    venv = Path(args.venv).resolve()
    llcpp = Path(args.llama_cpp).resolve()
    issues = diagnose(venv, llcpp, force_cpu=args.cpu,
                      check_service=not args.no_service_check)

    if args.json:
        print(json.dumps([i.to_dict() for i in issues], indent=2))
    else:
        gpu = GpuInfo.detect(force_cpu=args.cpu)
        venv_info = inspect_venv(venv)
        llcpp_info = inspect_llama_cpp(llcpp)
        print(f"venv:        {venv}  ({'OK' if venv_info.healthy else 'BROKEN'})")
        print(f"  py={venv_info.py_version}  torch={venv_info.torch_version}  "
              f"has_cuda={venv_info.torch_has_cuda}  "
              f"torchaudio={'ok' if venv_info.torchaudio_importable else 'BROKEN'} "
              f"({venv_info.torchaudio_version or '?'})")
        print(f"  fastapi={'ok' if venv_info.has_fastapi else 'MISSING'}  "
              f"llama_cpp={venv_info.llama_cpp_version or 'MISSING'}  "
              f"peft={venv_info.peft_version or 'MISSING'}  "
              f"trl={venv_info.trl_version or 'MISSING'}")
        print(f"GPU:         {gpu.name}  driver={gpu.driver_version}  "
              f"cuda_ver={gpu.cuda_ver or 'n/a'}  "
              f"toolkit={'yes' if gpu.cuda_toolkit_path else 'NO'}")
        print(f"llama.cpp:   dir={llcpp}  "
              f"quantize={'yes' if llcpp_info['quantize_exists'] else 'NO'}  "
              f"convert={'yes' if llcpp_info['convert_exists'] else 'NO'}")
        if not issues:
            print("issues:      none ✓")
        else:
            print(f"issues:      {len(issues)}")
            for i in issues:
                print(f"  [{i.severity}] {i.code}: {i.detail}")
                print(f"        fix: {i.suggested_fix}")

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

    if args.check:
        critical = any(i.severity >= 2 for i in issues)
        warn = any(i.severity == 1 for i in issues)
        return 2 if critical else (1 if warn else 0)

    # Default (diagnostic print only): exit non-zero on real issues so
    # `python install_diagnose.py && echo healthy` works in shell scripts.
    critical = any(i.severity >= 2 for i in issues)
    warn = any(i.severity == 1 for i in issues)
    return 2 if critical else (1 if warn else 0)


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
