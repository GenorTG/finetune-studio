"""Accelerator detection + install plan for Finetune Studio.

Single source of truth for every installer (`install.sh`, `update.sh`,
`install.ps1`, `install.bat`) and for `install_diagnose.py --repair`:

  detect()          -> GpuInfo      what hardware / driver stack is present
  build_plan(gpu)   -> Plan         which torch index, bitsandbytes, llama.cpp
                                    backend + CMake flags that hardware needs
  CLI               `plan`, `install`, `build-llama-cli`, `constraints`

Rules (Genor, 2026-10-05): CPU wheels are installed ONLY when no GPU of any
vendor is detected. A GPU whose driver stack is missing still gets the GPU
wheels plus a loud, actionable driver hint -- never a silent CPU fallback.
Detection never depends on one tool: NVIDIA (nvidia-smi, WSL path, sysfs),
AMD (rocm-smi, rocminfo, /opt/rocm, /dev/kfd, sysfs), Intel (xpu-smi, sycl-ls,
clinfo, sysfs), Apple (Darwin/arm64), plus lspci/sysfs for GPUs without
drivers. Stdlib only; safe to run with any system python3.

Test hooks: every probe goes through the module-level `_which/_run/_exists/
_read/_listdir` helpers, and `FTS_ACCEL_FIXTURE=<json>` swaps them for canned
fake hardware ({"which": [...], "commands": {"cmd args": "stdout"},
"files": {"/path": "text"}, "env": {...}, "system": "Linux", "machine": "x86_64"}).
`FTS_FORCE_VENDOR=nvidia|amd|intel|apple|none` and `--gpu` override detection.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

PYTORCH_WHL = "https://download.pytorch.org/whl"
ABETLEN_WHL = "https://abetlen.github.io/llama-cpp-python/whl"
PYPI_SIMPLE = "https://pypi.org/simple"
LLAMA_CPP_GIT = "https://github.com/ggerganov/llama.cpp"

# ── Index tables (verified against the live indexes 2026-10-05) ────────────
# (tag, (cuda_major, cuda_minor), lowest compute capability the wheel ships SASS for)
NVIDIA_TORCH_TAGS: tuple[tuple[str, tuple[int, int], float], ...] = (
    ("cu132", (13, 2), 7.5),
    ("cu130", (13, 0), 7.5),
    ("cu128", (12, 8), 7.0),   # 2.9+ cu128 builds dropped Maxwell/Pascal (cu126 keeps sm_50+)
    ("cu126", (12, 6), 5.0),
    ("cu124", (12, 4), 5.0),
    ("cu121", (12, 1), 5.0),
    ("cu118", (11, 8), 3.7),
)
# Blackwell (sm_100/sm_120) first appears in cu128 wheels.
BLACKWELL_MIN_TAG = (12, 8)
# Newest first. wheels bundle the ROCm userland, so only the kernel driver matters.
ROCM_TORCH_TAGS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("rocm7.2", (7, 2)), ("rocm7.1", (7, 1)), ("rocm7.0", (7, 0)),
    ("rocm6.4", (6, 4)), ("rocm6.3", (6, 3)), ("rocm6.2", (6, 2)),
)
# abetlen publishes CUDA wheels only for these tags (no cu126/cu128/cu129).
# Third field: lowest compute capability the wheel ships SASS for (cuobjdump 2026-10-05: cu132 is
# sm_75+, cu125 is sm_60+; nvcc 13 cannot target older SMs so cu130 follows cu132; 0.0 = not inspected).
ABETLEN_CUDA_TAGS: tuple[tuple[str, tuple[int, int], float], ...] = (
    ("cu132", (13, 2), 7.5), ("cu130", (13, 0), 7.5), ("cu125", (12, 5), 6.0),
    ("cu124", (12, 4), 0.0), ("cu123", (12, 3), 0.0), ("cu122", (12, 2), 0.0), ("cu121", (12, 1), 0.0),
)
# Lowest SM nvcc of a given major can still target.
NVCC_MIN_SM = {13: 75, 12: 50, 11: 35}

# Linux driver -> highest CUDA the driver supports (fallback when the
# nvidia-smi banner has no "CUDA Version"). Values from NVIDIA's release notes.
DRIVER_CUDA: tuple[tuple[int, tuple[int, int]], ...] = (
    (595, (13, 2)), (590, (13, 1)), (580, (13, 0)), (575, (12, 9)), (570, (12, 8)),
    (565, (12, 7)), (560, (12, 6)), (555, (12, 5)), (550, (12, 4)), (545, (12, 3)),
    (535, (12, 2)), (530, (12, 1)), (525, (12, 0)), (520, (11, 8)), (470, (11, 4)),
)

PCI_VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}
# Virtual / BMC / legacy VGA adapters that are not compute GPUs.
NOT_COMPUTE_GPU = re.compile(
    r"aspeed|matrox|vmware|qxl|virtio|cirrus|hyper-v|bochs|vga compatible controller: red hat|"
    r"parallels|virtualbox|\bmgag200|ast2[0-9]{3}|basic (display|render)", re.IGNORECASE)
# Intel graphics that torch XPU / SYCL actually support (Arc, Xe, Data Center, Core Ultra iGPU).
INTEL_XPU_NAMES = re.compile(
    r"arc|xe|battlemage|alchemist|celestial|data center gpu|flex|\bmax\b|core ultra|"
    r"meteor|lunar|arrow|panther|dg[12]|ponte", re.IGNORECASE)
# lspci chip codename -> CUDA compute capability (when no driver is loaded).
NVIDIA_CODENAMES: tuple[tuple[str, str], ...] = (
    (r"\bGB20[0-9]", "12.0"), (r"\bGB10[0-9]", "10.0"), (r"\bGH[0-9]{3}", "9.0"),
    (r"\bAD10[0-9]", "8.9"), (r"\bGA10[0-9]", "8.6"), (r"\bGA100", "8.0"),
    (r"\bTU1[0-9]{2}", "7.5"), (r"\bGV10[0-9]", "7.0"), (r"\bGP10[0-9]", "6.1"),
    (r"\bGP100", "6.0"), (r"\bGM[0-9]{3}", "5.2"),
)

# ── Probe helpers (module-level so tests / fixtures can swap them) ─────────

_FIXTURE: dict | None = None


def _fixture() -> dict | None:
    global _FIXTURE
    path = os.environ.get("FTS_ACCEL_FIXTURE")
    if not path:
        return None
    if _FIXTURE is None:
        _FIXTURE = json.loads(Path(path).read_text())
    return _FIXTURE


def _env(name: str, default: str = "") -> str:
    fx = _fixture()
    if fx is not None and name in fx.get("env", {}):
        return str(fx["env"][name])
    return os.environ.get(name, default)


def _system() -> tuple[str, str]:
    fx = _fixture()
    if fx is not None:
        return fx.get("system", "Linux"), fx.get("machine", "x86_64")
    return platform.system(), platform.machine()


def _which(cmd: str) -> str:
    fx = _fixture()
    if fx is not None:
        return f"/fixture/bin/{cmd}" if cmd in fx.get("which", []) else ""
    found = shutil.which(cmd)
    if found:
        return found
    if cmd == "nvidia-smi":  # WSL2 keeps it outside PATH
        wsl = "/usr/lib/wsl/lib/nvidia-smi"
        if Path(wsl).exists():
            return wsl
    return ""


def _run(cmd: Sequence[str], timeout: int = 15) -> tuple[int, str]:
    """Run a probe command. (127, "") when it cannot run at all."""
    fx = _fixture()
    if fx is not None:
        key = " ".join(cmd).replace("/fixture/bin/", "")
        known = fx.get("commands", {})
        match = key if key in known else max((k for k in known if key.startswith(k)), key=len, default="")
        if not match:
            return 127, ""
        v = known[match]
        if isinstance(v, dict):
            return int(v.get("rc", 0)), str(v.get("out", ""))
        return 0, str(v)
    try:
        r = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return 127, ""
    return r.returncode, (r.stdout or "") + ("" if r.returncode == 0 else (r.stderr or ""))


def _exists(path: str) -> bool:
    fx = _fixture()
    if fx is not None:
        p = path.rstrip("/")
        return any(k == p or k.startswith(p + "/") for k in fx.get("files", {}))
    return Path(path).exists()


def _read(path: str) -> str:
    fx = _fixture()
    if fx is not None:
        return str(fx.get("files", {}).get(path, ""))
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError:
        return ""


def _listdir(path: str) -> list[str]:
    fx = _fixture()
    if fx is not None:
        p = path.rstrip("/") + "/"
        return sorted({k[len(p):].split("/")[0] for k in fx.get("files", {}) if k.startswith(p)})
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def _ver(text: str) -> tuple[int, int]:
    m = re.search(r"(\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _vstr(v: tuple[int, int]) -> str:
    return f"{v[0]}.{v[1]}" if v != (0, 0) else ""


# ── Data model ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Gpu:
    vendor: str
    name: str
    compute_cap: str = ""   # NVIDIA "8.6"
    gfx: str = ""           # AMD "gfx1100"
    index: int = -1


@dataclass(frozen=True)
class GpuInfo:
    """Detected accelerator. The first six fields are the legacy positional API."""
    vendor: str            # "nvidia" | "amd" | "intel" | "apple" | "none"
    name: str
    driver_version: str
    cuda_ver: str          # torch tag: "cu132" / "cu126" / ... ("" when not NVIDIA)
    compute_cap: str       # newest visible GPU: "12.0" Blackwell, "8.6" Ampere
    cuda_toolkit_path: str
    gpus: tuple[Gpu, ...] = ()
    cuda_max: str = ""         # highest CUDA the driver supports ("13.0")
    nvcc: str = ""             # newest usable nvcc
    nvcc_version: str = ""
    rocm_version: str = ""
    hipcc: str = ""
    icpx: str = ""
    vulkan_ready: bool = False  # glslc + vulkan headers present
    driver_ready: bool = True   # False: hardware seen but no working driver stack
    hint: str = ""              # actionable driver / toolchain hint
    notes: tuple[str, ...] = ()

    @classmethod
    def detect(cls, force_cpu: bool = False, force_vendor: str = "") -> GpuInfo:
        return detect(force_cpu=force_cpu, force_vendor=force_vendor)

    @property
    def compute_caps(self) -> list[str]:
        return sorted({g.compute_cap for g in self.gpus if g.compute_cap})


# ── NVIDIA ───────────────────────────────────────────────────────────────

def _cuda_for_driver(driver: str) -> tuple[int, int]:
    try:
        major = int(driver.split(".")[0])
    except ValueError:
        return (0, 0)
    for floor, cuda in DRIVER_CUDA:
        if major >= floor:
            return cuda
    return (0, 0)


def _visible_filter(gpus: list[Gpu]) -> list[Gpu]:
    """Honor CUDA_VISIBLE_DEVICES indices (UUID lists / empty = ignore)."""
    raw = _env("CUDA_VISIBLE_DEVICES").strip()
    if not raw or not all(p.strip().isdigit() for p in raw.split(",")):
        return gpus
    want = {int(p) for p in raw.split(",")}
    kept = [g for g in gpus if g.index in want]
    return kept or gpus


def _probe_nvidia() -> tuple[list[Gpu], str, tuple[int, int]]:
    smi = _which("nvidia-smi")
    if not smi:
        return [], "", (0, 0)
    rc, out = _run([smi, "--query-gpu=index,name,driver_version,compute_cap",
                    "--format=csv,noheader"])
    if rc != 0 or not out.strip():  # old drivers lack compute_cap
        rc, out = _run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"])
    gpus: list[Gpu] = []
    driver = ""
    for i, line in enumerate(out.strip().splitlines() if rc == 0 else []):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            idx = int(parts[0]) if parts[0].isdigit() else i
            name, driver, cc = parts[1], parts[2], parts[3]
        elif len(parts) >= 2:
            idx, name, driver, cc = i, parts[0], parts[1], ""
        else:
            continue
        gpus.append(Gpu("nvidia", name, cc if re.fullmatch(r"\d+\.\d+", cc) else "", index=idx))
    cuda_max = (0, 0)
    if gpus:
        rc, banner = _run([smi])
        m = re.search(r"CUDA (?:UMD )?Version:\s*(\d+\.\d+)", banner) if rc == 0 else None
        cuda_max = _ver(m.group(1)) if m else _cuda_for_driver(driver)
    return gpus, driver, cuda_max


def _cc_from_name(name: str) -> str:
    for pat, cc in NVIDIA_CODENAMES:
        if re.search(pat, name):
            return cc
    return ""


# ── Toolchains ───────────────────────────────────────────────────────────

def _nvcc_targets(version: tuple[int, int], cc: float) -> bool:
    """Can an nvcc of this release generate code for compute capability `cc`?"""
    if cc >= 10 and version < BLACKWELL_MIN_TAG:   # sm_100/sm_120 arrived with CUDA 12.8
        return False
    return cc * 10 >= NVCC_MIN_SM.get(version[0], 50)


def _find_nvcc(cuda_max: tuple[int, int], ccs: Sequence[float] = ()) -> tuple[str, str]:
    """Best nvcc: one that can target every visible GPU first (a CUDA 12.x toolkit next to
    nvcc 13 for a Pascal card), then the driver's CUDA major, else the newest <= it."""
    cands: list[str] = []
    for src in (_env("CUDACXX"), _which("nvcc")):
        if src:
            cands.append(src)
    for base in ["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda"] + [
            f"/usr/local/{d}" for d in _listdir("/usr/local") if d.startswith("cuda-")]:
        cands.append(f"{base}/bin/nvcc")
    seen: dict[str, tuple[int, int]] = {}
    for c in cands:
        if c in seen or not _exists(c):
            continue
        rc, out = _run([c, "--version"])
        m = re.search(r"release (\d+\.\d+)", out)
        if rc == 0 and m:
            seen[c] = _ver(m.group(1))
    if not seen:
        return "", ""
    want = cuda_max[0]
    ranked = sorted(seen.items(),
                    key=lambda kv: (all(_nvcc_targets(kv[1], c) for c in ccs), kv[1][0] == want,
                                    kv[1][0] <= want or want == 0, kv[1]),
                    reverse=True)
    best = ranked[0]
    return best[0], _vstr(best[1])


def _vulkan_ready() -> bool:
    return bool(_which("glslc")) and (_exists("/usr/include/vulkan/vulkan.h")
                                      or _exists("/usr/local/include/vulkan/vulkan.h")
                                      or bool(_which("pkg-config")) and _run(["pkg-config", "--exists", "vulkan"])[0] == 0)


# ── PCI scan (works with no vendor tool installed) ─────────────────────────

def _scan_pci() -> list[Gpu]:
    """Display/3D controllers on the PCI bus, with or without any vendor driver or tool.
    lspci gives names (and chip codenames -> NVIDIA compute capability); sysfs is the no-lspci fallback."""
    gpus = [g for g in _scan_lspci_names() if not NOT_COMPUTE_GPU.search(g.name)] if _which("lspci") else []
    if gpus:
        return gpus
    base = "/sys/bus/pci/devices"
    for dev in _listdir(base):
        cls = _read(f"{base}/{dev}/class")
        vendor = PCI_VENDORS.get(_read(f"{base}/{dev}/vendor").lower())
        if vendor and cls.startswith(("0x03", "0x0b40")):  # display / 3D / co-processor
            gpus.append(Gpu(vendor, f"{vendor} GPU [{dev}]"))
    return gpus


def _scan_windows() -> list[Gpu]:
    """Windows has no sysfs/lspci: ask WMI for the adapter names (finds driverless GPUs too)."""
    rc, out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_VideoController).Name -join \"`n\""])
    gpus: list[Gpu] = []
    for line in out.splitlines() if rc == 0 else []:
        low = line.lower()
        if NOT_COMPUTE_GPU.search(line) or "microsoft basic" in low:
            continue
        vendor = ("nvidia" if "nvidia" in low else
                  "amd" if re.search(r"\bamd\b|radeon|advanced micro", low) else
                  "intel" if "intel" in low else "")
        if vendor:
            gpus.append(Gpu(vendor, line.strip(), compute_cap=_cc_from_name(line) if vendor == "nvidia" else ""))
    return gpus


def _scan_lspci_names() -> list[Gpu]:
    rc, out = _run(["lspci", "-nn"])
    res: list[Gpu] = []
    for line in out.splitlines() if rc == 0 else []:
        if not re.search(r"\[03(00|02|80)\]|VGA compatible|3D controller|Display controller", line):
            continue
        low = line.lower()
        vendor = ("nvidia" if "nvidia" in low else
                  "amd" if re.search(r"\bamd\b|advanced micro devices|\bati\b", low) else
                  "intel" if "intel" in low else "")
        if vendor:
            name = line.split(": ", 1)[-1].strip()
            res.append(Gpu(vendor, name, compute_cap=_cc_from_name(name) if vendor == "nvidia" else ""))
    return res


# ── AMD / Intel ──────────────────────────────────────────────────────────

def _probe_amd() -> tuple[list[Gpu], str, str]:
    """(gpus, rocm_version, hipcc)"""
    gpus: list[Gpu] = []
    rocm_ver = ""
    for f in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        if _exists(f):
            rocm_ver = _vstr(_ver(_read(f)))
            break
    hipcc = next((p for p in (_which("hipcc"), "/opt/rocm/bin/hipcc") if p and _exists(p)), "")
    gfx_names: list[str] = []
    for tool in ("rocminfo", "/opt/rocm/bin/rocminfo"):
        if _which(tool) or _exists(tool):
            rc, out = _run([_which(tool) or tool])
            if rc == 0:
                gfx_names = [g for g in re.findall(r"\b(gfx[0-9a-f]{3,4})\b", out) if g != "gfx000"]
                break
    for node in _listdir("/sys/class/kfd/kfd/topology/nodes"):  # driver loaded, no userland
        m = re.search(r"gfx_target_version\s+(\d+)", _read(f"/sys/class/kfd/kfd/topology/nodes/{node}/properties"))
        if m and int(m.group(1)) > 0:
            v = int(m.group(1))
            gfx_names.append(f"gfx{v // 10000}{(v // 100) % 100:x}{v % 100:x}")
    names: list[str] = []
    if _which("rocm-smi"):
        rc, out = _run(["rocm-smi", "--showproductname"])
        names = [m.strip() for m in re.findall(r"Card (?:series|model):\s*(.+)", out)] if rc == 0 else []
    seen_gfx = list(dict.fromkeys(gfx_names))
    for i, g in enumerate(seen_gfx or [""] * len(names)):
        gpus.append(Gpu("amd", names[i] if i < len(names) else "AMD GPU (ROCm)", gfx=g, index=i))
    if not gpus and (_which("rocm-smi") or _exists("/opt/rocm") or _exists("/dev/kfd")):
        gpus.append(Gpu("amd", "AMD GPU (ROCm)", index=0))
    return gpus, rocm_ver, hipcc


def _probe_intel() -> tuple[list[Gpu], str]:
    """(gpus, icpx path). Only GPU devices count -- sycl-ls also lists the CPU."""
    gpus: list[Gpu] = []
    icpx = next((p for p in (_which("icpx"), "/opt/intel/oneapi/compiler/latest/bin/icpx") if p and _exists(p)), "")
    if _which("sycl-ls"):
        rc, out = _run(["sycl-ls"])
        for line in out.splitlines() if rc == 0 else []:
            if re.search(r"\[(level_zero|opencl):gpu\]", line) and "Intel" in line:
                gpus.append(Gpu("intel", line.split("]", 2)[-1].strip(), index=len(gpus)))
    if not gpus and _which("xpu-smi"):
        rc, out = _run(["xpu-smi", "discovery"])
        for m in re.finditer(r"Device Name:\s*(.+)", out) if rc == 0 else []:
            gpus.append(Gpu("intel", m.group(1).strip(), index=len(gpus)))
    if not gpus and _which("clinfo"):
        rc, out = _run(["clinfo", "-l"])
        for line in out.splitlines() if rc == 0 else []:
            if "Intel" in line and re.search(r"Graphics|Arc|Xe", line):
                gpus.append(Gpu("intel", line.strip(" `-"), index=len(gpus)))
    return gpus, icpx


# ── Detection ─────────────────────────────────────────────────────────────

def _cuda_path() -> str:
    return next((c for c in ("/opt/cuda", "/usr/local/cuda", "/usr/lib/cuda") if _exists(c)), "")


def detect(force_cpu: bool = False, force_vendor: str = "") -> GpuInfo:
    cuda_path = _cuda_path()
    if force_cpu:
        return GpuInfo("none", "(forced CPU)", "", "", "", cuda_path)
    force_vendor = force_vendor or _env("FTS_FORCE_VENDOR")
    system, machine = _system()
    notes: list[str] = []

    nv_gpus, driver, cuda_max = _probe_nvidia()
    nv_gpus = _visible_filter(nv_gpus)
    amd_gpus, rocm_ver, hipcc = _probe_amd()
    intel_all, icpx = _probe_intel()
    pci = _scan_pci() if system == "Linux" else _scan_windows() if system == "Windows" else []

    apple = system == "Darwin" and machine in ("arm64", "aarch64")
    # PCI-visible hardware whose userland tool is missing/broken (driverless GPUs).
    pci_by = {v: [g for g in pci if g.vendor == v] for v in ("nvidia", "amd", "intel")}
    intel_gpus = intel_all or pci_by["intel"]
    intel_supported = [g for g in intel_gpus if INTEL_XPU_NAMES.search(g.name) or g.name.startswith("intel GPU [")]
    amd_hw = amd_gpus or pci_by["amd"]
    nv_hw = nv_gpus or pci_by["nvidia"]
    if intel_gpus and not intel_supported:
        notes.append("Intel integrated graphics detected (UHD/HD generation): PyTorch XPU does not support it.")

    vendor = ""
    if force_vendor in ("nvidia", "amd", "intel", "apple", "none"):
        vendor = force_vendor
    elif nv_hw:
        vendor = "nvidia"
    elif amd_hw:
        vendor = "amd"
    elif intel_supported:
        vendor = "intel"
    elif apple:
        vendor = "apple"
    else:
        vendor = "none"

    vulkan = _vulkan_ready()
    common = {"cuda_toolkit_path": cuda_path, "vulkan_ready": vulkan, "hipcc": hipcc, "icpx": icpx}

    if vendor == "nvidia":
        gpus = nv_hw or [Gpu("nvidia", "NVIDIA GPU")]
        ready = bool(nv_gpus)
        hint = ""
        if not ready:
            cuda_max = cuda_max if cuda_max != (0, 0) else (13, 0)
            hint = ("NVIDIA GPU found on the PCI bus but no working driver (nvidia-smi missing/failing). "
                    "Install the vendor driver, then re-run `bash install.sh --repair`: "
                    "Debian/Ubuntu `sudo apt install nvidia-driver` (or `ubuntu-drivers install`), "
                    "Fedora `sudo dnf install akmod-nvidia`, Arch `sudo pacman -S nvidia-open`. "
                    "CUDA wheels are installed anyway.")
        ccs = [float(g.compute_cap) for g in gpus if g.compute_cap]
        newest = max(ccs) if ccs else 0.0
        nvcc, nvcc_v = _find_nvcc(cuda_max, ccs)
        tag = pick_cuda_tag(cuda_max, newest)
        for g in gpus:
            if g.compute_cap and tag and float(g.compute_cap) < _tag_floor(tag):
                notes.append(f"GPU{f' {g.index}' if g.index >= 0 else ''} ({g.name}, sm_{g.compute_cap.replace('.', '')}) is too old for "
                             f"torch {tag} wheels; PyTorch will skip it (use CUDA_VISIBLE_DEVICES).")
        if tag and BLACKWELL_MIN_TAG > _tag_ver(tag) and newest >= 10:
            notes.append("Blackwell GPU needs a CUDA 12.8+ driver (>=570).")
        return GpuInfo("nvidia", gpus[0].name, driver, tag, f"{newest:.1f}" if newest else "",
                       cuda_path, tuple(gpus), _vstr(cuda_max), nvcc, nvcc_v,
                       driver_ready=ready, hint=hint, notes=tuple(notes), **{
                           k: v for k, v in common.items() if k != "cuda_toolkit_path"})

    if vendor == "amd":
        gpus = amd_hw or [Gpu("amd", "AMD GPU (ROCm)")]
        ready = bool(amd_gpus) and (_exists("/dev/kfd") or bool(_which("rocm-smi")))
        hint = "" if ready else (
            "AMD GPU found but the ROCm kernel/user stack is not usable (no /dev/kfd, rocm-smi, rocminfo). "
            "Install ROCm (https://rocm.docs.amd.com/projects/install-on-linux): "
            "`sudo apt install amdgpu-dkms rocm` then add your user to the render+video groups and reboot. "
            "ROCm PyTorch wheels are installed anyway.")
        return GpuInfo("amd", gpus[0].name, "", "", "", cuda_path, tuple(gpus), "", "", "",
                       rocm_version=rocm_ver, driver_ready=ready, hint=hint, notes=tuple(notes),
                       **{k: v for k, v in common.items() if k != "cuda_toolkit_path"})

    if vendor == "intel":
        gpus = intel_supported or intel_gpus or [Gpu("intel", "Intel XPU")]
        ready = bool(intel_all)
        hint = "" if ready else (
            "Intel GPU found but the compute runtime is missing (no sycl-ls/xpu-smi/clinfo device). "
            "Install intel-compute-runtime + level-zero: Debian/Ubuntu "
            "`sudo apt install intel-opencl-icd libze-intel-gpu1 libze1` "
            "(https://dgpu-docs.intel.com), add your user to the render group, reboot. "
            "XPU PyTorch wheels are installed anyway.")
        return GpuInfo("intel", gpus[0].name, "", "", "", cuda_path, tuple(gpus), "", "", "",
                       driver_ready=ready, hint=hint, notes=tuple(notes),
                       **{k: v for k, v in common.items() if k != "cuda_toolkit_path"})

    if vendor == "apple":
        return GpuInfo("apple", "Apple Silicon (Metal/MPS)", "", "", "", cuda_path,
                       (Gpu("apple", "Apple Silicon"),), notes=tuple(notes), vulkan_ready=False)

    return GpuInfo("none", "(no GPU)", "", "", "", cuda_path, notes=tuple(notes), vulkan_ready=vulkan)


# ── Selection ────────────────────────────────────────────────────────────

def _tag_ver(tag: str) -> tuple[int, int]:
    return next((v for t, v, _ in NVIDIA_TORCH_TAGS if t == tag), (0, 0))


def _tag_floor(tag: str) -> float:
    return next((f for t, _, f in NVIDIA_TORCH_TAGS if t == tag), 0.0)


def pick_cuda_tag(cuda_max: tuple[int, int], newest_cc: float = 0.0) -> str:
    """Newest torch CUDA wheel index the driver can run and the newest GPU supports.

    CUDA 13.x drivers run any cu13x wheel (minor-version compatibility, verified
    cu132 on a 580/CUDA 13.0 driver); 12.x drivers only get wheels <= their own
    version (newer 12.x wheels on old drivers fail with missing-symbol errors)."""
    if cuda_max == (0, 0):
        cuda_max = (13, 0)  # unknown driver: assume a current one
    for tag, ver, floor in NVIDIA_TORCH_TAGS:
        if ver[0] >= 13 and cuda_max[0] >= 13:
            ok = True
        elif ver[0] == 11 and cuda_max[0] >= 11:
            ok = True      # 11.x wheels run on any 11.x+ driver (minor-version compatibility)
        else:
            ok = ver <= cuda_max
        if not ok:
            continue
        if newest_cc and newest_cc >= 10 and ver < BLACKWELL_MIN_TAG:
            continue
        if newest_cc and newest_cc < floor:
            continue
        return tag
    # Nothing satisfies driver + GPU (e.g. Blackwell on a pre-12.8 driver): hand back the
    # oldest tag that can run the GPU so install proceeds and the note tells the user to upgrade.
    if newest_cc >= 10:
        return "cu128"
    return "cu118"


def _rocm_floor(gfx: str) -> tuple[int, int]:
    """Oldest ROCm torch wheel with kernels for this GPU: RDNA4 (gfx12xx) needs 6.4+."""
    return (6, 4) if re.match(r"gfx12", gfx) else (0, 0)


def pick_rocm_tag(rocm_version: str, gfx: str = "") -> str:
    installed = _ver(rocm_version)
    floor = _rocm_floor(gfx)
    for tag, ver in ROCM_TORCH_TAGS:
        if (installed == (0, 0) or ver <= installed) and ver >= floor:
            return tag
    # installed ROCm is older than anything this GPU/the index list supports: take the
    # oldest index that satisfies the GPU floor (wheels bundle their own ROCm userland).
    fit = [t for t, v in ROCM_TORCH_TAGS if v >= floor]
    return fit[-1] if fit else ROCM_TORCH_TAGS[-1][0]


def cuda_archs(gpu: GpuInfo) -> str:
    """CMAKE_CUDA_ARCHITECTURES for the visible GPUs, limited to what nvcc can target."""
    nvcc_major = _ver(gpu.nvcc_version)[0] if gpu.nvcc_version else 0
    floor = NVCC_MIN_SM.get(nvcc_major, 50)
    sms = sorted({int(float(c) * 10) for c in gpu.compute_caps if float(c) * 10 >= floor})
    if sms:
        return ";".join(str(s) for s in sms)
    # No known compute capability: let nvcc resolve `native`. Known GPUs that nvcc cannot target return ""
    # (`native` would resolve to the very SM nvcc rejects); `nvcc_problem` explains it to the user.
    return "" if gpu.compute_caps else "native"


def nvcc_problem(gpu: GpuInfo) -> str:
    """Why the discovered nvcc cannot build llama.cpp for the visible GPUs ('' = it can)."""
    if not gpu.nvcc_version or not gpu.compute_caps:
        return ""
    ver = _ver(gpu.nvcc_version)
    ccs = [float(c) for c in gpu.compute_caps]
    if any(c >= 10 for c in ccs) and ver < BLACKWELL_MIN_TAG:
        sm = "sm_" + str(int(max(ccs) * 10))
        return (f"nvcc {gpu.nvcc_version} cannot target Blackwell ({sm}): it needs CUDA 12.8 or newer. "
                "Install a current toolkit (https://developer.nvidia.com/cuda-downloads; Debian/Ubuntu apt "
                "toolkits are 11.8/12.0) and re-run `bash install.sh --repair`.")
    if not cuda_archs(gpu):
        sm = "sm_" + str(int(min(ccs) * 10))
        return (f"nvcc {gpu.nvcc_version} cannot target {sm} (CUDA {ver[0]} dropped that architecture): install an "
                "older CUDA 12.x toolkit next to it (e.g. `cuda-toolkit-12-6`) and re-run `bash install.sh --repair`.")
    return ""


@dataclass
class LlamaBackend:
    name: str                  # cuda | hip | sycl | vulkan | metal | cpu
    cmake_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    env_prefix: str = ""       # shell snippet to source first (oneAPI setvars)
    missing: str = ""          # toolchain hint when this wanted backend is unbuildable


def llama_backends(gpu: GpuInfo) -> list[LlamaBackend]:
    """Backends in preference order for this hardware; unbuildable ones carry `missing`."""
    vulkan = LlamaBackend("vulkan", ["-DGGML_VULKAN=ON"],
                          missing="" if gpu.vulkan_ready else
                          "Vulkan toolchain missing: sudo apt install libvulkan-dev glslc (or vulkan-sdk)")
    cpu = LlamaBackend("cpu")
    out: list[LlamaBackend] = []
    if gpu.vendor == "nvidia":
        env, miss = {}, nvcc_problem(gpu)
        args = ["-DGGML_CUDA=ON"]
        if archs := cuda_archs(gpu):
            args.append(f"-DCMAKE_CUDA_ARCHITECTURES={archs}")
        if gpu.nvcc:
            home = str(Path(gpu.nvcc).resolve().parent.parent)
            env = {"CUDACXX": gpu.nvcc, "CUDA_HOME": home, "CUDA_PATH": home}
            args.append(f"-DCMAKE_CUDA_COMPILER={gpu.nvcc}")
        else:
            miss = ("nvcc not found. Install the CUDA toolkit matching your driver "
                    f"(CUDA {gpu.cuda_max or '13.x'}): https://developer.nvidia.com/cuda-downloads "
                    "(Debian/Ubuntu: `sudo apt install nvidia-cuda-toolkit` or NVIDIA's cuda-toolkit-13-x)")
        out.append(LlamaBackend("cuda", args, env, missing=miss))
    elif gpu.vendor == "amd":
        env, miss = {}, ""
        args = ["-DGGML_HIP=ON", "-DCMAKE_BUILD_TYPE=Release"]
        gfx = ";".join(sorted({g.gfx for g in gpu.gpus if g.gfx}))
        if gfx:
            args.append(f"-DGPU_TARGETS={gfx}")
        if gpu.hipcc:
            root = str(Path(gpu.hipcc).resolve().parent.parent)
            env = {"HIPCXX": f"{root}/llvm/bin/clang", "HIP_PATH": root}
        else:
            miss = ("hipcc not found. Install the ROCm HIP SDK: `sudo apt install rocm-hip-sdk` "
                    "(https://rocm.docs.amd.com/projects/install-on-linux)")
        out.append(LlamaBackend("hip", args, env, missing=miss))
    elif gpu.vendor == "intel":
        setvars = "/opt/intel/oneapi/setvars.sh"
        miss = "" if (gpu.icpx or _exists(setvars)) else (
            "Intel oneAPI DPC++ (icpx) not found. Install intel-oneapi-compiler-dpcpp-cpp + intel-oneapi-mkl-devel "
            "(https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html)")
        out.append(LlamaBackend(
            "sycl", ["-DGGML_SYCL=ON", "-DCMAKE_C_COMPILER=icx", "-DCMAKE_CXX_COMPILER=icpx"],
            env_prefix=f". {setvars} >/dev/null 2>&1" if _exists(setvars) else "", missing=miss))
    elif gpu.vendor == "apple":
        return [LlamaBackend("metal", ["-DGGML_METAL=ON"])]
    if gpu.vendor in ("nvidia", "amd", "intel"):
        out.append(vulkan)
    out.append(cpu)
    return out


@dataclass
class Plan:
    vendor: str
    gpu_name: str
    driver: str
    torch_tag: str                  # cu132 / rocm7.2 / xpu / cpu / "" (mac default)
    torch_index: str                # full URL, "" => default PyPI (mac)
    torch_backend: str              # cuda | hip | xpu | mps | cpu  (what torch must report)
    torch_fallback_tags: list[str]  # tried in order if the first index fails/verify fails
    bnb: bool
    unsloth_ok: bool
    llama_backends: list[LlamaBackend]
    wheel_index: str                # abetlen prebuilt-wheel index ("" => none for this hw)
    warnings: list[str]
    info: GpuInfo


def build_plan(gpu: GpuInfo, *, unsloth: str = "") -> Plan:
    """Turn detected hardware into concrete install choices."""
    warn = list(gpu.notes)
    if gpu.hint:
        warn.append(gpu.hint)
    fallbacks: list[str] = []
    wheel = ""
    if gpu.vendor == "nvidia":
        tag = gpu.cuda_ver or "cu126"
        newest = float(gpu.compute_cap) if gpu.compute_cap else 0.0
        fallbacks = [t for t, v, f in NVIDIA_TORCH_TAGS
                     if _tag_ver(t) < _tag_ver(tag) and (newest == 0 or newest >= f)
                     and not (newest >= 10 and v < BLACKWELL_MIN_TAG)]
        idx, backend = f"{PYTORCH_WHL}/{tag}", "cuda"
        # Prebuilt llama-cpp-python wheel: needs the CUDA runtime libs on the box,
        # a tag abetlen publishes, and kernels for the GPU (sm_120 has none).
        if newest < 12:
            wheel = pick_abetlen_wheel(gpu)
    elif gpu.vendor == "amd" and _system()[0] == "Windows":
        # There are no ROCm PyTorch wheels for Windows: say so loudly instead of failing the install.
        tag, idx, backend = "cpu", f"{PYTORCH_WHL}/cpu", "cpu"
        warn.append("AMD GPU on Windows: PyTorch ROCm wheels are Linux-only, so training falls back to CPU "
                    "wheels here. For GPU training use WSL2 + ROCm, or Linux. llama.cpp can still use Vulkan.")
    elif gpu.vendor == "amd":
        gfx = next((g.gfx for g in gpu.gpus if g.gfx), "")
        tag = pick_rocm_tag(gpu.rocm_version, gfx)
        floor = _rocm_floor(gfx)   # a fallback below the GPU's floor would install "fine" with no kernels
        fallbacks = [t for t, v in ROCM_TORCH_TAGS
                     if t != tag and floor <= v < _ver(tag.replace("rocm", ""))]
        idx, backend = f"{PYTORCH_WHL}/{tag}", "hip"
    elif gpu.vendor == "intel":
        tag, idx, backend = "xpu", f"{PYTORCH_WHL}/xpu", "xpu"
    elif gpu.vendor == "apple":
        tag, idx, backend, wheel = "", "", "mps", f"{ABETLEN_WHL}/metal"
    else:
        tag, idx, backend = "cpu", f"{PYTORCH_WHL}/cpu", "cpu"
        if gpu.name == "(forced CPU)":
            warn.append("CPU wheels forced (--cpu / FTS_FORCE_VENDOR=none).")
        else:
            warn.append("No GPU detected (checked nvidia-smi, rocm-smi/rocminfo, xpu-smi/sycl-ls/clinfo, "
                        "sysfs PCI, lspci): installing CPU wheels. Re-run after installing GPU drivers.")
    # unsloth: NVIDIA CUDA only (AMD/Intel/mac paths are not supported by its kernels here)
    want = (unsloth or _env("FTS_UNSLOTH", "auto")).lower()
    unsloth_ok = gpu.vendor == "nvidia" and want not in ("0", "off", "no", "false")
    return Plan(gpu.vendor, gpu.name, gpu.driver_version, tag, idx, backend, fallbacks,
                bnb=gpu.vendor in ("nvidia", "amd", "intel", "apple"), unsloth_ok=unsloth_ok,
                llama_backends=llama_backends(gpu), wheel_index=wheel, warnings=warn, info=gpu)


def pick_abetlen_wheel(gpu: GpuInfo) -> str:
    """Prebuilt llama-cpp-python wheel index the driver can run, whose runtime libs are on the box
    and whose kernels cover the OLDEST visible GPU ("" => source build).

    A CUDA 13 driver also runs cu12x wheels, so a Pascal/Volta GPU that the cu13x wheels have no
    kernels for gets cu125 instead of a wheel that installs, passes the offload probe and then
    fails with "no kernel image is available" on the first load."""
    cm = _ver(gpu.cuda_max)
    if cm == (0, 0):
        return ""
    ccs = [float(c) for c in gpu.compute_caps]
    oldest = min(ccs) if ccs else 0.0
    for tag, v, floor in ABETLEN_CUDA_TAGS:
        if v[0] > cm[0] or (v[0] == cm[0] and cm[0] < 13 and v > cm) or oldest < floor:
            continue
        if _has_cuda_runtime(gpu, v[0]):
            return f"{ABETLEN_WHL}/{tag}"
    return ""


def _has_cuda_runtime(gpu: GpuInfo, major: int) -> bool:
    """libcudart.so.<major> loadable on the system (abetlen CUDA wheels link it dynamically)."""
    if not major:
        return False
    if _system()[0] != "Linux":
        return True   # Windows wheels are self-contained / resolve DLLs from the CUDA install
    rc, out = _run(["ldconfig", "-p"])
    if rc == 0 and f"libcudart.so.{major}" in out:
        return True
    roots = [gpu.cuda_toolkit_path] if gpu.cuda_toolkit_path else []
    return any(_exists(f"{r}/targets/x86_64-linux/lib/libcudart.so.{major}") or
               _exists(f"{r}/lib64/libcudart.so.{major}") for r in roots)


def best_llama_backend(plan: Plan) -> tuple[LlamaBackend, list[str]]:
    """First buildable backend + loud messages for each preferred one that is not."""
    msgs: list[str] = []
    for b in plan.llama_backends:
        if b.missing:
            msgs.append(f"llama.cpp {b.name.upper()} backend unavailable: {b.missing}")
            continue
        if b.name == "cpu" and plan.vendor != "none":
            msgs.append("llama.cpp will be CPU-only (no GPU backend toolchain found) -- GGUF "
                        "inference/export will not use your GPU until the hints above are fixed; "
                        "then run `bash install.sh --repair`.")
        return b, msgs
    return plan.llama_backends[-1], msgs


# ── Constraints file (keeps torch on its GPU build across later resolves) ──

def constraint_lines(versions: dict[str, str]) -> list[str]:
    lines = [f"{n}=={versions[n]}" for n in ("torch", "torchvision", "torchaudio") if versions.get(n)]
    tv = versions.get("torch", "")
    try:
        major, minor = (int(p) for p in tv.split("+")[0].split(".")[:2])
        if (major, minor) < (2, 7):
            # torchao >=0.17 calls torch.utils._pytree.register_constant (torch 2.7+) and
            # transformers 5.x imports torchao eagerly; unsloth_zoo hard-requires it.
            lines.append("torchao<0.17")
    except ValueError:
        pass
    return lines


def write_constraints(venv: Path) -> Path | None:
    py = venv / ("Scripts/python.exe" if (venv / "Scripts/python.exe").exists() else "bin/python")
    code = ("import importlib.metadata as m, json\nout={}\n"
            "for n in ('torch','torchvision','torchaudio'):\n"
            "    try: out[n]=m.version(n)\n    except m.PackageNotFoundError: pass\nprint(json.dumps(out))")
    rc, out = _run([str(py), "-c", code], timeout=60)
    if rc != 0:
        return None
    try:
        lines = constraint_lines(json.loads(out.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        return None
    if not lines:
        return None
    path = venv / "torch-constraints.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


# ── Command builders (pure; executed by `install` / repair) ────────────────

def uv_pip(py: str, *args: str) -> list[str]:
    return ["uv", "pip", "install", "--python", py, *args]


def constraint_args(venv: Path) -> list[str]:
    f = venv / "torch-constraints.txt"
    return ["-c", str(f)] if f.exists() and f.stat().st_size else []


def torch_specs(plan: Plan, unsloth_profile: bool = False) -> list[str]:
    if unsloth_profile:  # unsloth currently pins torch<2.13
        return ["torch<2.13", "torchvision"]
    return ["torch", "torchvision"]  # torchaudio is frozen at 2.11 and unused by the app


def torch_commands(plan: Plan, py: str, *, unsloth_profile: bool = False) -> list[tuple[str, list[str]]]:
    """(tag, command) attempts, preferred first. CPU only when the plan itself is CPU."""
    specs = torch_specs(plan, unsloth_profile)
    base = ["--reinstall-package", "torch", "--reinstall-package", "torchvision"]
    if plan.torch_index == "":  # macOS: default PyPI wheels carry MPS
        return [("pypi", uv_pip(py, *base, *specs))]
    tags = [plan.torch_tag] + plan.torch_fallback_tags
    prefix = f"{PYTORCH_WHL}/"
    return [(t, uv_pip(py, *base, "--index-url", prefix + t, *specs)) for t in tags]


def torch_probe_code(backend: str) -> str:
    want = {"cuda": "torch.version.cuda", "hip": "torch.version.hip",
            "xpu": "bool(getattr(torch.version, 'xpu', None)) or '+xpu' in torch.__version__",
            "mps": "torch.backends.mps.is_built()", "cpu": "True"}[backend]
    return f"import torch; print(torch.__version__); raise SystemExit(0 if ({want}) else 3)"


def llama_py_attempts(plan: Plan, py: str, constraints: list[str]) -> list[tuple[str, list[str], dict[str, str]]]:
    """(label, uv command, env) llama-cpp-python attempts, GPU-capable ones first."""
    spec = "llama-cpp-python>=0.3.0"
    force = ["--reinstall-package", "llama-cpp-python"]
    attempts: list[tuple[str, list[str], dict[str, str]]] = []
    if plan.wheel_index:
        # uv searches --extra-index-url BEFORE the default index (PyPI); first-index then takes
        # llama-cpp-python from the GPU wheel index and everything else from PyPI. Putting the wheel
        # index in --index-url instead makes PyPI win and silently installs the CPU sdist.
        attempts.append((f"prebuilt wheel {plan.wheel_index.rsplit('/', 1)[-1]}",
                         uv_pip(py, *force, *constraints, "--extra-index-url", plan.wheel_index,
                                "--index-strategy", "first-index", spec), {}))
    for b in plan.llama_backends:
        if b.name == "cpu" or b.missing:
            continue
        env = {"CMAKE_ARGS": " ".join(b.cmake_args), "FORCE_CMAKE": "1", **b.env}
        if b.name == "hip" and b.env.get("HIPCXX"):
            env.update({"CC": b.env["HIPCXX"], "CXX": b.env["HIPCXX"] + "++"})
        # --no-cache: uv keys its built-wheel cache on the sdist only, NOT on CMAKE_ARGS, so a cached
        # CPU build would be reused and the GPU flags silently ignored (verified 2026-10-05).
        attempts.append((f"source build ({b.name})",
                         uv_pip(py, "--no-cache", *force, *constraints, "--no-binary", "llama-cpp-python", spec), env))
    attempts.append(("CPU build", uv_pip(py, *force, *constraints, spec), {}))
    return attempts


LLAMA_MARKER_NAME = "fts-llama-cpp-python.json"


def write_llama_marker(venv: Path, plan: Plan, label: str) -> None:
    """Record how llama-cpp-python was installed (diagnose uses it to tell a Blackwell
    source build from a prebuilt wheel that lacks sm_120 kernels)."""
    best, _ = best_llama_backend(plan)
    arch = next((a.split("=", 1)[1] for a in best.cmake_args
                 if a.startswith("-DCMAKE_CUDA_ARCHITECTURES=")), "")
    data = {"label": label, "backend": best.name, "source": label.startswith("source"), "arch": arch}
    try:
        (venv / LLAMA_MARKER_NAME).write_text(json.dumps(data))
    except OSError:
        pass


LLAMA_PROBE = ("import llama_cpp; print(llama_cpp.__version__); "
               "raise SystemExit(0 if llama_cpp.llama_supports_gpu_offload() else 4)")


def llama_cli_dirs(root: Path) -> tuple[Path, Path]:
    return root / "convert_hf_to_gguf.py", root / "build" / "bin" / "llama-quantize"


def detect_llama_cli_backend(root: Path) -> str:
    """Backend an existing .llama.cpp build was configured for ('' unknown)."""
    marker = root / "build" / ".fts-backend"
    if marker.exists():
        return marker.read_text().strip()
    cache = root / "build" / "CMakeCache.txt"
    text = cache.read_text(errors="replace") if cache.exists() else ""
    for flag, name in (("GGML_CUDA", "cuda"), ("GGML_HIP", "hip"), ("GGML_SYCL", "sycl"),
                       ("GGML_VULKAN", "vulkan"), ("GGML_METAL", "metal")):
        if re.search(rf"^{flag}:BOOL=ON", text, re.MULTILINE):
            return name
    return "cpu" if text else ""


# ── Executors ────────────────────────────────────────────────────────────

def _say(msg: str) -> None:
    print(f"[accel] {msg}", flush=True)


def _exec(cmd: list[str], env: dict[str, str] | None = None, timeout: int = 3600,
          shell_prefix: str = "") -> int:
    full_env = {**os.environ, **(env or {})}
    quiet = {"capture_output": True} if "--dry-run" in cmd else {}   # resolver probes: we print our own verdict
    try:
        if shell_prefix:
            r = subprocess.run(["bash", "-c", f'{shell_prefix}; exec "$@"', "_", *cmd], env=full_env,
                               timeout=timeout, check=False, **quiet)
        else:
            r = subprocess.run(cmd, env=full_env, timeout=timeout, check=False, **quiet)
    except (OSError, subprocess.SubprocessError) as exc:
        _say(f"cannot run {cmd[0]}: {exc}")
        return 127
    return r.returncode


def venv_python(venv: Path) -> str:
    win = venv / "Scripts" / "python.exe"
    return str(win if win.exists() else venv / "bin" / "python")


def install_torch(plan: Plan, venv: Path, *, unsloth_profile: bool = False, runner=_exec) -> bool:
    py = venv_python(venv)
    for tag, cmd in torch_commands(plan, py, unsloth_profile=unsloth_profile):
        _say(f"Installing PyTorch ({tag or 'default'}) for {plan.vendor}...")
        if runner(cmd) != 0:
            _say(f"torch install from {tag} failed -- trying the next compatible index")
            continue
        if runner([py, "-c", torch_probe_code(plan.torch_backend)]) == 0:
            return True
        _say(f"torch from {tag} is not a {plan.torch_backend} build -- trying the next index")
    _say("ERROR: no PyTorch index produced a working build "
         f"(wanted {plan.torch_backend}). NOT falling back to CPU wheels on a GPU host.")
    return False


def install_llama_cpp_python(plan: Plan, venv: Path, *, runner=_exec) -> bool:
    py = venv_python(venv)
    attempts = llama_py_attempts(plan, py, constraint_args(venv))
    gpu_expected = plan.vendor != "none"
    for label, cmd, env in attempts:
        if label == "CPU build" and gpu_expected:
            _say("WARN: no GPU build of llama-cpp-python could be installed; installing the "
                 "CPU build so GGUF still works. Fix the hints above, then `bash install.sh --repair`.")
        _say(f"llama-cpp-python: {label}")
        if runner(cmd, env) != 0:
            continue
        probe = runner([py, "-c", LLAMA_PROBE])
        if probe == 0 or label == "CPU build" or not gpu_expected:
            write_llama_marker(venv, plan, label)
            return True
        _say(f"llama-cpp-python ({label}) installed but has no GPU offload -- trying the next option")
    return False


def install_bitsandbytes(plan: Plan, venv: Path, *, runner=_exec) -> bool:
    if not plan.bnb:
        return True
    return runner(uv_pip(venv_python(venv), *constraint_args(venv), "bitsandbytes>=0.45.5")) == 0


def stack_constraints_text(venv: Path) -> str:
    """Pins of the core training stack as installed (so unsloth cannot downgrade them)."""
    code = ("import importlib.metadata as m\nfor n in ('torch','torchvision','transformers','trl',"
            "'peft','accelerate','datasets','huggingface-hub','tokenizers'):\n"
            "    try: print(n+'=='+m.version(n))\n    except m.PackageNotFoundError: pass")
    rc, out = _run([venv_python(venv), "-c", code], timeout=60)
    return out if rc == 0 else ""


UNSLOTH_FLOOR = "unsloth>=2026.1.0"   # without a floor uv backtracks to 2025.x + protobuf 3.x and breaks onnxruntime


def install_unsloth(plan: Plan, venv: Path, *, profile: bool = False, runner=_exec) -> bool:
    """Optional, NVIDIA only. Default: install only if a release resolves against the installed
    torch/transformers/trl/datasets pinned (else skip, never downgrade the core stack). Profile
    (FTS_UNSLOTH=1): torch<2.13 was installed on purpose, so let unsloth pick the older
    transformers/trl it needs."""
    if not plan.unsloth_ok:
        return True
    py = venv_python(venv)
    if profile:
        return runner(uv_pip(py, *constraint_args(venv), UNSLOTH_FLOOR)) == 0
    cfile = venv / "stack-constraints.txt"
    cfile.write_text(stack_constraints_text(venv))
    cmd = uv_pip(py, "-c", str(cfile), UNSLOTH_FLOOR)
    if runner(cmd + ["--dry-run"]) != 0:
        _say("unsloth: skipped -- no unsloth release is compatible with the installed torch/transformers/trl "
             "(unsloth pins torch<2.13, transformers<=5.5, trl<=0.24). Training uses the standard TRL path. "
             "For unsloth: FTS_UNSLOTH=1 bash install.sh (builds a torch<2.13 stack).")
        return True
    return runner(cmd) == 0


def build_llama_cli(plan: Plan, root: Path, venv: Path, *, rebuild: bool = False,
                    only: bool = False, runner=_exec) -> bool:
    """Clone + build llama.cpp (convert_hf_to_gguf.py, llama-quantize, llama-cli) with the
    best available backend, falling back down the list (never silently to CPU on a GPU host)."""
    convert, quantize = llama_cli_dirs(root)
    want, msgs = best_llama_backend(plan)
    for m in msgs:
        _say(f"WARN: {m}")
    if convert.exists() and quantize.exists() and not rebuild:
        have = detect_llama_cli_backend(root)
        if have in ("", want.name) or want.name == "cpu":
            _say(f"llama.cpp CLI present at {root} (backend {have or 'unknown'}).")
            return True
        _say(f"llama.cpp CLI at {root} was built for '{have}' but this host needs '{want.name}' -- rebuilding")
    if not shutil.which("cmake") or not shutil.which("git"):
        _say("ERROR: cmake and git are required to build the llama.cpp CLI.")
        return False
    if not root.exists() and runner(["git", "clone", "--depth", "1", LLAMA_CPP_GIT, str(root)]) != 0:
        _say("ERROR: git clone llama.cpp failed")
        return False
    if not only:  # the convert script's pinned torch must not displace our GPU build
        req = root / "requirements" / "requirements-convert_hf_to_gguf.txt"
        runner(uv_pip(venv_python(venv), "--quiet", *constraint_args(venv), "-r", str(req)))
    build = root / "build"
    chosen = ""
    ordered = [b for b in plan.llama_backends if not b.missing]
    for b in ordered:
        if b.name == "cpu" and plan.vendor != "none":
            _say("WARN: building llama.cpp CPU-only on a GPU host (see hints above).")
        if build.exists():
            shutil.rmtree(build)
        _say(f"llama.cpp: configuring backend {b.name} {' '.join(b.cmake_args)}")
        cfg = ["cmake", "-S", str(root), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
               "-DCMAKE_RULE_MESSAGES=OFF", "-DLLAMA_BUILD_TESTS=OFF", *b.cmake_args]
        if runner(cfg, b.env, shell_prefix=b.env_prefix) != 0:
            _say(f"llama.cpp {b.name} configure failed -- trying the next backend")
            continue
        if runner(["cmake", "--build", str(build), "--config", "Release", "-j"], b.env,
                  shell_prefix=b.env_prefix) != 0 or not quantize.exists():
            _say(f"llama.cpp {b.name} build failed -- trying the next backend")
            continue
        chosen = b.name
        break
    if not chosen:
        _say("ERROR: llama.cpp build failed for every backend")
        return False
    (build / ".fts-backend").write_text(chosen + "\n")
    _say(f"llama.cpp CLI built with backend '{chosen}': {quantize}")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────

def _shell_vars(plan: Plan) -> str:
    best, msgs = best_llama_backend(plan)
    g = plan.info
    d = {
        "GPU_VENDOR": plan.vendor, "GPU_NAME": plan.gpu_name, "GPU_DRIVER": plan.driver,
        "CUDA_VER": g.cuda_ver, "CUDA_MAX": g.cuda_max, "COMPUTE_CAP": g.compute_cap,
        "TORCH_TAG": plan.torch_tag, "TORCH_INDEX": plan.torch_index, "TORCH_BACKEND": plan.torch_backend,
        "LLAMA_BACKEND": best.name, "LLAMA_CMAKE_ARGS": " ".join(best.cmake_args),
        "LLAMA_WHEEL_INDEX": plan.wheel_index, "NVCC": g.nvcc, "ROCM_VERSION": g.rocm_version,
        "DRIVER_READY": "1" if g.driver_ready else "0",
        "PLAN_WARNINGS": "\n".join(plan.warnings + msgs),
    }
    return "\n".join(f"{k}={shlex.quote(str(v))}" for k, v in d.items())


def plan_to_dict(plan: Plan) -> dict:
    best, msgs = best_llama_backend(plan)
    g = plan.info
    return {
        "vendor": plan.vendor, "gpu_name": plan.gpu_name, "driver": plan.driver,
        "driver_ready": g.driver_ready, "cuda_max": g.cuda_max, "compute_caps": g.compute_caps,
        "torch_tag": plan.torch_tag, "torch_index": plan.torch_index, "torch_backend": plan.torch_backend,
        "torch_fallback_tags": plan.torch_fallback_tags, "bitsandbytes": plan.bnb,
        "unsloth": plan.unsloth_ok, "llama_backend": best.name,
        "llama_cmake_args": best.cmake_args, "llama_wheel_index": plan.wheel_index,
        "llama_backends": [{"name": b.name, "missing": b.missing} for b in plan.llama_backends],
        "nvcc": g.nvcc, "nvcc_version": g.nvcc_version, "rocm_version": g.rocm_version,
        "warnings": plan.warnings + msgs,
    }


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Finetune Studio accelerator plan + installer helpers.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "install", "build-llama-cli", "constraints"):
        p = sub.add_parser(name)
        p.add_argument("--cpu", action="store_true", help="force CPU wheels")
        p.add_argument("--gpu", default="", choices=["", "nvidia", "amd", "intel", "apple", "none"],
                       help="override detection")
        p.add_argument("--venv", default=".venv")
        if name == "plan":
            p.add_argument("--shell", action="store_true", help="emit KEY='value' lines for eval")
        if name == "install":
            p.add_argument("what", choices=["torch", "llama-cpp-python", "bitsandbytes", "unsloth", "all"])
            p.add_argument("--dry-run", action="store_true", help="print commands only")
        if name == "build-llama-cli":
            p.add_argument("--dir", default=str(Path(__file__).resolve().parent.parent / ".llama.cpp"))
            p.add_argument("--rebuild", action="store_true")
            p.add_argument("--only", action="store_true", help="skip the convert-script python deps")
    a = ap.parse_args(argv)
    gpu = detect(force_cpu=a.cpu, force_vendor=a.gpu)
    plan = build_plan(gpu)
    venv = Path(a.venv).resolve()

    if a.cmd == "plan":
        print(_shell_vars(plan) if a.shell else json.dumps(plan_to_dict(plan), indent=2))
        return 0
    if a.cmd == "constraints":
        path = write_constraints(venv)
        if path:
            print(path)
        return 0 if path else 1
    if a.cmd == "build-llama-cli":
        return 0 if build_llama_cli(plan, Path(a.dir), venv, rebuild=a.rebuild or _env("FTS_REBUILD_LLAMA_CPP") == "1",
                                    only=a.only) else 1

    unsloth_profile = _env("FTS_UNSLOTH").lower() in ("1", "on", "yes", "true", "force")
    plan.unsloth_ok = plan.unsloth_ok or (unsloth_profile and plan.vendor == "nvidia")

    def runner(cmd: list[str], env: dict[str, str] | None = None, shell_prefix: str = "") -> int:
        if a.dry_run:
            print("DRY-RUN:", " ".join(shlex.quote(c) for c in cmd), ("ENV " + json.dumps(env)) if env else "")
            return 0
        return _exec(cmd, env, shell_prefix=shell_prefix)

    ok = True
    if a.what in ("torch", "all"):
        ok = install_torch(plan, venv, unsloth_profile=unsloth_profile, runner=runner) and ok
        if ok and not a.dry_run:
            write_constraints(venv)
    if a.what in ("llama-cpp-python", "all") and ok:
        ok = install_llama_cpp_python(plan, venv, runner=runner) and ok
    if a.what in ("bitsandbytes", "all") and ok:
        ok = install_bitsandbytes(plan, venv, runner=runner) and ok
    if a.what in ("unsloth", "all") and ok:
        ok = install_unsloth(plan, venv, profile=unsloth_profile, runner=runner) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
