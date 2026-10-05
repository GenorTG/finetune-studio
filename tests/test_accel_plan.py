"""Accelerator detection + install plan (scripts/accel_plan.py), driven by fake hardware.

Every probe in accel_plan goes through swappable helpers; these tests feed it
canned fixtures (`FTS_ACCEL_FIXTURE`) so NVIDIA / AMD / Intel / Apple /
driverless / CPU-only hosts are all exercised on any machine, hermetically.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import accel_plan as ap

SMI_QUERY = "nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv,noheader"
NVCC13 = "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.4, V13.4.92\n"
NVCC118 = "Cuda compilation tools, release 11.8, V11.8.89\n"


def banner(cuda: str, label: str = "CUDA UMD Version") -> str:
    return f"| NVIDIA-SMI 580.1  Driver Version: x   {label}: {cuda} |\n"


def fake_hw(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **spec: object) -> None:
    spec.setdefault("system", "Linux")
    spec.setdefault("machine", "x86_64")
    path = tmp_path / "hw.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setenv("FTS_ACCEL_FIXTURE", str(path))
    monkeypatch.delenv("FTS_FORCE_VENDOR", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(ap, "_FIXTURE", None)


def nvidia(gpus: list[tuple[str, str]], driver: str, cuda: str, *, nvcc: bool = True, runtime: bool = True) -> dict:
    lines = "".join(f"{i}, {name}, {driver}, {cc}\n" for i, (name, cc) in enumerate(gpus))
    files = {"/usr/local/cuda/bin/nvcc": "x"} if nvcc else {}
    if runtime:
        files["/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.13"] = "x"
        files["/usr/local/cuda/lib64/libcudart.so.12"] = "x"
    return {
        "which": ["nvidia-smi", "nvcc"] if nvcc else ["nvidia-smi"],
        "commands": {SMI_QUERY: lines, "nvidia-smi": banner(cuda),
                     "/usr/local/cuda/bin/nvcc --version": NVCC13, "nvcc --version": NVCC13},
        "files": files,
    }


R3090 = ("NVIDIA GeForce RTX 3090", "8.6")
G1070 = ("NVIDIA GeForce GTX 1070", "6.1")
R5090 = ("NVIDIA GeForce RTX 5090", "12.0")


# ── NVIDIA: newest wheel index the DRIVER supports ─────────────────────────

@pytest.mark.parametrize("driver,cuda,tag", [
    ("580.178.04", "13.0", "cu132"),   # CUDA 13 drivers run any cu13x wheel (verified live on 580)
    ("595.10", "13.2", "cu132"),
    ("575.51", "12.9", "cu128"),
    ("570.26", "12.8", "cu128"),
    ("560.28", "12.6", "cu126"),
    ("550.54", "12.4", "cu124"),
    ("535.54", "12.2", "cu121"),
    ("525.60", "12.0", "cu118"),       # 12.0 driver: no cu120 wheels exist
])
def test_nvidia_picks_newest_index_the_driver_supports(monkeypatch, tmp_path, driver, cuda, tag) -> None:
    fake_hw(monkeypatch, tmp_path, **nvidia([R3090], driver, cuda))
    g = ap.detect()
    assert (g.vendor, g.cuda_ver, g.compute_cap, g.cuda_max) == ("nvidia", tag, "8.6", cuda)
    plan = ap.build_plan(g)
    assert plan.torch_index == f"{ap.PYTORCH_WHL}/{tag}" and plan.torch_backend == "cuda"
    assert "cpu" not in [plan.torch_tag, *plan.torch_fallback_tags]


def test_driver_version_alone_is_enough_when_banner_has_no_cuda_line(monkeypatch, tmp_path) -> None:
    spec = nvidia([R3090], "570.26", "12.8")
    spec["commands"]["nvidia-smi"] = "garbage\n"
    fake_hw(monkeypatch, tmp_path, **spec)
    assert ap.detect().cuda_ver == "cu128"


def test_old_driver_without_compute_cap_query_still_detects(monkeypatch, tmp_path) -> None:
    spec = nvidia([R3090], "470.42", "11.4")
    spec["commands"][SMI_QUERY] = {"rc": 2, "out": "Field compute_cap is not a valid field"}
    spec["commands"]["nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"] = "RTX 3090, 470.42\n"
    fake_hw(monkeypatch, tmp_path, **spec)
    g = ap.detect()
    assert g.vendor == "nvidia" and g.driver_version == "470.42" and g.compute_cap == ""
    assert g.cuda_ver == "cu118"


def test_blackwell_never_gets_a_pre_cu128_wheel(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, **nvidia([R5090], "570.26", "12.8"))
    g = ap.detect()
    assert g.cuda_ver == "cu128" and g.compute_cap == "12.0"
    plan = ap.build_plan(g)
    assert plan.torch_fallback_tags == []          # cu126 and older cannot run sm_120
    assert plan.wheel_index == ""                  # abetlen wheels lack sm_120 kernels -> source build
    cuda = next(b for b in plan.llama_backends if b.name == "cuda")
    assert "-DCMAKE_CUDA_ARCHITECTURES=120" in cuda.cmake_args


def test_pascal_only_host_falls_back_to_cu126_which_still_ships_sm_60(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, **nvidia([G1070], "580.178.04", "13.0"))
    assert ap.detect().cuda_ver == "cu126"


def test_mixed_host_targets_the_newest_gpu_and_warns_about_the_unsupported_one(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, **nvidia([R3090, G1070], "580.178.04", "13.0"))
    g = ap.detect()
    assert g.cuda_ver == "cu132"
    assert any("GPU 1" in n and "GTX 1070" in n for n in g.notes)
    # llama.cpp CUDA arch list drops sm_61: nvcc 13 cannot target it
    assert ap.cuda_archs(g) == "86"


def test_cuda_visible_devices_restricts_which_gpus_count(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, **nvidia([R3090, G1070], "580.178.04", "13.0"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert ap.detect().cuda_ver == "cu126"           # only the 1070 is visible
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    g = ap.detect()
    assert g.cuda_ver == "cu132" and not g.notes


def test_empty_cuda_visible_devices_does_not_turn_a_gpu_host_into_cpu(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, **nvidia([R3090], "580.178.04", "13.0"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert ap.detect().vendor == "nvidia"


def test_nvcc_choice_prefers_the_driver_major_over_an_ancient_path_nvcc(monkeypatch, tmp_path) -> None:
    spec = nvidia([R3090], "580.178.04", "13.0")
    spec["commands"]["nvcc --version"] = NVCC118           # PATH nvcc is CUDA 11.8 (this very box)
    spec["files"]["/usr/local/cuda-13.4/bin/nvcc"] = "x"
    spec["commands"]["/usr/local/cuda-13.4/bin/nvcc --version"] = NVCC13
    fake_hw(monkeypatch, tmp_path, **spec)
    g = ap.detect()
    assert g.nvcc.endswith("cuda-13.4/bin/nvcc") or g.nvcc.endswith("cuda/bin/nvcc")
    assert g.nvcc_version == "13.4"


# ── GPU present but no driver: warn + GPU wheels, never silent CPU ──────────

def test_nvidia_hardware_without_driver_gets_cuda_wheels_and_a_driver_hint(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, files={
        "/sys/bus/pci/devices/0000:01:00.0/vendor": "0x10de",
        "/sys/bus/pci/devices/0000:01:00.0/class": "0x030000",
    })
    g = ap.detect()
    assert g.vendor == "nvidia" and g.driver_ready is False
    assert "driver" in g.hint.lower() and "install" in g.hint.lower()
    plan = ap.build_plan(g)
    assert plan.torch_backend == "cuda" and plan.torch_tag.startswith("cu") and plan.torch_tag != "cpu"
    assert any("driver" in w.lower() for w in plan.warnings)


def test_lspci_codename_gives_compute_capability_when_no_driver(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, which=["lspci"], commands={
        "lspci -nn": "01:00.0 VGA compatible controller [0300]: NVIDIA Corporation GB202 [GeForce RTX 5090] [10de:2b85]\n"})
    g = ap.detect()
    assert g.vendor == "nvidia" and g.compute_cap == "12.0" and g.cuda_ver == "cu132"


def test_amd_hardware_without_rocm_gets_rocm_wheels_and_hint(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, files={
        "/sys/bus/pci/devices/0000:03:00.0/vendor": "0x1002",
        "/sys/bus/pci/devices/0000:03:00.0/class": "0x030000"})
    g = ap.detect()
    assert (g.vendor, g.driver_ready) == ("amd", False)
    assert "ROCm" in g.hint
    plan = ap.build_plan(g)
    assert plan.torch_tag == "rocm7.2" and plan.torch_backend == "hip"


def test_virtual_and_bmc_adapters_are_not_gpus(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, which=["lspci"], commands={
        "lspci -nn": "03:00.0 VGA compatible controller [0300]: ASPEED Technology, Inc. ASPEED Graphics Family [1a03:2000]\n"
                     "00:02.0 VGA compatible controller [0300]: VMware SVGA II Adapter [15ad:0405]\n"})
    assert ap.detect().vendor == "none"


# ── AMD ROCm ───────────────────────────────────────────────────────────────

def amd(rocm: str = "6.4.1", gfx: str = "gfx1100", hipcc: bool = True) -> dict:
    files = {"/opt/rocm/.info/version": rocm, "/dev/kfd": ""}
    if hipcc:
        files["/opt/rocm/bin/hipcc"] = "x"
    return {"which": ["rocm-smi", "rocminfo"], "files": files,
            "commands": {"rocminfo": f"Name: {gfx}\nName: AMD Ryzen CPU\n",
                         "rocm-smi --showproductname": "Card series: Radeon RX 7900 XTX\n"}}


@pytest.mark.parametrize("rocm,tag", [("7.2.0", "rocm7.2"), ("7.1.1", "rocm7.1"), ("6.4.1", "rocm6.4"),
                                      ("6.2.4", "rocm6.2"), ("", "rocm7.2")])
def test_amd_torch_index_follows_the_installed_rocm(monkeypatch, tmp_path, rocm, tag) -> None:
    spec = amd(rocm=rocm or "x")
    if not rocm:
        spec["files"].pop("/opt/rocm/.info/version")
    fake_hw(monkeypatch, tmp_path, **spec)
    g = ap.detect()
    assert g.vendor == "amd" and g.driver_ready
    assert ap.build_plan(g).torch_tag == tag


def test_rdna4_needs_rocm_6_4_or_newer() -> None:
    assert ap.pick_rocm_tag("6.3.0", "gfx1201") == "rocm6.4"


def test_amd_llama_backend_is_hip_with_gpu_targets_else_vulkan(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, **amd())
    plan = ap.build_plan(ap.detect())
    best, msgs = ap.best_llama_backend(plan)
    assert best.name == "hip" and "-DGGML_HIP=ON" in best.cmake_args and "-DGPU_TARGETS=gfx1100" in best.cmake_args
    assert best.env["HIP_PATH"] == "/opt/rocm" and not msgs
    # no hipcc, no vulkan SDK -> CPU, loudly
    fake_hw(monkeypatch, tmp_path, **amd(hipcc=False))
    best, msgs = ap.best_llama_backend(ap.build_plan(ap.detect()))
    assert best.name == "cpu" and any("hipcc" in m for m in msgs) and any("CPU-only" in m for m in msgs)
    # glslc + headers present -> Vulkan instead of CPU
    spec = amd(hipcc=False)
    spec["which"].append("glslc")
    spec["files"]["/usr/include/vulkan/vulkan.h"] = "x"
    fake_hw(monkeypatch, tmp_path, **spec)
    best, _ = ap.best_llama_backend(ap.build_plan(ap.detect()))
    assert best.name == "vulkan" and best.cmake_args == ["-DGGML_VULKAN=ON"]


# ── Intel XPU ──────────────────────────────────────────────────────────────

def test_intel_arc_gets_xpu_wheels_and_sycl(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, which=["sycl-ls"], files={"/opt/intel/oneapi/setvars.sh": "x",
            "/opt/intel/oneapi/compiler/latest/bin/icpx": "x"},
            commands={"sycl-ls": "[opencl:cpu][opencl:0] Intel(R) Xeon(R)\n"
                                 "[level_zero:gpu][level_zero:0] Intel(R) Arc(TM) A770 Graphics 12.55\n"})
    g = ap.detect()
    plan = ap.build_plan(g)
    assert g.vendor == "intel" and g.driver_ready
    assert (plan.torch_tag, plan.torch_index, plan.torch_backend) == ("xpu", f"{ap.PYTORCH_WHL}/xpu", "xpu")
    best, _ = ap.best_llama_backend(plan)
    assert best.name == "sycl" and "-DGGML_SYCL=ON" in best.cmake_args and "setvars.sh" in best.env_prefix


def test_sycl_ls_cpu_only_is_not_a_gpu(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, which=["sycl-ls"],
            commands={"sycl-ls": "[opencl:cpu][opencl:0] Intel(R) Core(TM) i7 CPU\n"})
    assert ap.detect().vendor == "none"


def test_intel_uhd_igpu_is_noted_and_does_not_pull_xpu_wheels(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, which=["lspci"], commands={
        "lspci -nn": "00:02.0 VGA compatible controller [0300]: Intel Corporation UHD Graphics 630 [8086:3e92]\n"})
    g = ap.detect()
    assert g.vendor == "none" and any("UHD" in n or "integrated" in n for n in g.notes)


def test_intel_xe_without_runtime_gets_xpu_wheels_with_a_runtime_hint(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, which=["lspci"], commands={
        "lspci -nn": "03:00.0 VGA compatible controller [0300]: Intel Corporation DG2 [Arc A750] [8086:56a1]\n"})
    g = ap.detect()
    assert (g.vendor, g.driver_ready) == ("intel", False)
    assert "compute-runtime" in g.hint
    assert ap.build_plan(g).torch_tag == "xpu"


# ── Apple / CPU ────────────────────────────────────────────────────────────

def test_apple_silicon_uses_default_wheels_mps_and_metal(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, system="Darwin", machine="arm64")
    plan = ap.build_plan(ap.detect())
    assert (plan.vendor, plan.torch_index, plan.torch_backend) == ("apple", "", "mps")
    assert plan.wheel_index.endswith("/whl/metal")
    assert ap.best_llama_backend(plan)[0].cmake_args == ["-DGGML_METAL=ON"]
    cmds = ap.torch_commands(plan, "py")
    assert len(cmds) == 1 and "--index-url" not in cmds[0][1]


def test_cpu_wheels_only_when_no_gpu_of_any_vendor(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path)
    plan = ap.build_plan(ap.detect())
    assert (plan.vendor, plan.torch_tag, plan.torch_backend) == ("none", "cpu", "cpu")
    assert plan.torch_index.endswith("/whl/cpu") and plan.bnb is False
    assert any("No GPU detected" in w for w in plan.warnings)
    assert [t for t, _ in ap.torch_commands(plan, "py")] == ["cpu"]


def test_force_cpu_and_force_vendor(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, **nvidia([R3090], "580.1", "13.0"))
    forced = ap.build_plan(ap.detect(force_cpu=True))
    assert forced.torch_tag == "cpu" and any("forced" in w for w in forced.warnings)
    monkeypatch.setenv("FTS_FORCE_VENDOR", "amd")
    assert ap.detect().vendor == "amd"


# ── llama-cpp-python / llama.cpp backends ──────────────────────────────────

def test_nvidia_llama_cpp_python_prefers_prebuilt_wheel_then_source_then_cpu(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, **nvidia([R3090], "580.178.04", "13.0"))
    plan = ap.build_plan(ap.detect())
    labels = [a[0] for a in ap.llama_py_attempts(plan, "py", [])]
    assert labels == ["prebuilt wheel cu132", "source build (cuda)", "CPU build"]
    wheel = ap.llama_py_attempts(plan, "py", [])[0][1]
    # uv searches --extra-index-url BEFORE PyPI; with first-index llama-cpp-python comes from abetlen and
    # deps from PyPI. Using --index-url for abetlen lets PyPI win and silently installs the CPU sdist
    # (verified 2026-10-05).
    assert wheel[wheel.index("--extra-index-url") + 1] == f"{ap.ABETLEN_WHL}/cu132"
    assert "first-index" in wheel and "--index-url" not in wheel
    src_env = ap.llama_py_attempts(plan, "py", [])[1][2]
    assert "-DGGML_CUDA=ON" in src_env["CMAKE_ARGS"] and "-DCMAKE_CUDA_ARCHITECTURES=86" in src_env["CMAKE_ARGS"]
    assert src_env["FORCE_CMAKE"] == "1" and src_env["CUDACXX"].endswith("nvcc")


def test_no_cuda_runtime_libs_means_no_prebuilt_wheel_attempt(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, **nvidia([R3090], "580.178.04", "13.0", runtime=False))
    assert ap.build_plan(ap.detect()).wheel_index == ""


def test_cuda12_driver_uses_a_cu12x_abetlen_wheel(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, **nvidia([R3090], "550.54", "12.4"))
    assert ap.build_plan(ap.detect()).wheel_index.endswith("/whl/cu124")


def test_missing_nvcc_is_reported_and_best_backend_is_not_silent(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, **nvidia([R3090], "580.178.04", "13.0", nvcc=False, runtime=False))
    plan = ap.build_plan(ap.detect())
    best, msgs = ap.best_llama_backend(plan)
    assert best.name == "cpu"
    assert any("nvcc" in m and "CUDA" in m for m in msgs) and any("CPU-only" in m for m in msgs)


@pytest.mark.parametrize("backend,has", [("cpu", False), ("cuda", True)])
def test_llama_cpp_python_cpu_build_is_last_resort_and_flagged_on_gpu_hosts(backend, has) -> None:
    gpu = ap.GpuInfo("nvidia", "x", "580", "cu132", "8.6", "/usr/local/cuda", gpus=(ap.Gpu("nvidia", "x", "8.6", index=0),),
                     cuda_max="13.0", nvcc="/usr/local/cuda/bin/nvcc", nvcc_version="13.4")
    plan = ap.build_plan(gpu)
    ran: list[list[str]] = []
    said: list[str] = []

    def runner(cmd, env=None, shell_prefix=""):
        ran.append(cmd)
        return 4 if "llama_cpp" in " ".join(cmd) and backend == "cpu" else 0   # probe: no offload

    ap._say = said.append  # type: ignore[assignment]
    try:
        ok = ap.install_llama_cpp_python(plan, Path("/nonexistent-venv"), runner=runner)
    finally:
        ap._say = lambda m: print(f"[accel] {m}", flush=True)  # type: ignore[assignment]
    assert ok
    installs = [c for c in ran if "install" in c]
    if has:
        assert len(installs) == 1                       # wheel installed AND probe passed
    else:
        assert len(installs) == 3 and "--no-binary" in installs[1]
        assert any("no GPU build of llama-cpp-python" in m for m in said)   # loud, never silent


def test_cuda_archs_are_limited_to_what_nvcc_can_target() -> None:
    caps = (ap.Gpu("nvidia", "a", "6.1", index=0), ap.Gpu("nvidia", "b", "8.6", index=1), ap.Gpu("nvidia", "c", "12.0", index=2))
    new = ap.GpuInfo("nvidia", "a", "580", "cu132", "12.0", "", gpus=caps, nvcc_version="13.4")
    old = ap.GpuInfo("nvidia", "a", "535", "cu121", "12.0", "", gpus=caps, nvcc_version="12.2")
    assert ap.cuda_archs(new) == "86;120"
    assert ap.cuda_archs(old) == "61;86;120"
    assert ap.cuda_archs(ap.GpuInfo("nvidia", "a", "", "", "", "")) == "native"


# ── Install execution (fake runner: no network) ────────────────────────────

def _plan_for(monkeypatch, tmp_path, **spec) -> ap.Plan:
    fake_hw(monkeypatch, tmp_path, **spec)
    return ap.build_plan(ap.detect())


def test_torch_install_steps_down_indexes_but_never_to_cpu_on_a_gpu_host(monkeypatch, tmp_path) -> None:
    plan = _plan_for(monkeypatch, tmp_path, **nvidia([R3090], "580.178.04", "13.0"))
    seen: list[str] = []

    def runner(cmd, env=None, shell_prefix=""):
        seen.append(" ".join(cmd))
        return 1  # every index "fails"

    assert ap.install_torch(plan, tmp_path, runner=runner) is False
    assert any("/cu132" in c for c in seen) and any("/cu126" in c for c in seen)
    assert not any("/whl/cpu" in c for c in seen)


def test_torch_install_rejects_a_wheel_of_the_wrong_backend_and_tries_the_next(monkeypatch, tmp_path) -> None:
    plan = _plan_for(monkeypatch, tmp_path, **nvidia([R3090], "580.178.04", "13.0"))
    calls: list[list[str]] = []

    def runner(cmd, env=None, shell_prefix=""):
        calls.append(cmd)
        is_probe = cmd[1:2] == ["-c"]
        return 3 if is_probe and len([c for c in calls if c[1:2] == ["-c"]]) == 1 else 0

    assert ap.install_torch(plan, tmp_path, runner=runner) is True
    installs = [c for c in calls if "install" in c]
    assert len(installs) == 2 and installs[1][installs[1].index("--index-url") + 1].endswith("/cu130")


def test_torch_family_is_torch_and_torchvision_only_no_frozen_torchaudio() -> None:
    plan = ap.build_plan(ap.GpuInfo("none", "(no GPU)", "", "", "", ""))
    assert ap.torch_specs(plan) == ["torch", "torchvision"]
    assert ap.torch_specs(plan, unsloth_profile=True)[0] == "torch<2.13"


def test_constraints_pin_the_torch_family_and_cap_torchao_below_torch_2_7() -> None:
    assert ap.constraint_lines({"torch": "2.14.1+cu132", "torchvision": "0.29.1+cu132"}) == [
        "torch==2.14.1+cu132", "torchvision==0.29.1+cu132"]
    assert "torchao<0.17" in ap.constraint_lines({"torch": "2.6.0+cu124"})


def test_bitsandbytes_is_installed_for_every_gpu_vendor_and_skipped_for_cpu(monkeypatch, tmp_path) -> None:
    for spec, want in (({}, False), (amd(), True), (nvidia([R3090], "580.1", "13.0"), True)):
        plan = _plan_for(monkeypatch, tmp_path, **spec)
        ran: list[list[str]] = []
        assert ap.install_bitsandbytes(plan, tmp_path, runner=lambda c, e=None, shell_prefix="", r=ran: r.append(c) or 0)
        assert bool(ran) is want


def test_unsloth_is_skipped_not_forced_when_it_cannot_resolve_against_the_stack(monkeypatch, tmp_path) -> None:
    plan = _plan_for(monkeypatch, tmp_path, **nvidia([R3090], "580.1", "13.0"))
    venv = tmp_path / "v"
    venv.mkdir()
    ran: list[list[str]] = []

    def runner(cmd, env=None, shell_prefix=""):
        ran.append(cmd)
        return 1 if "--dry-run" in cmd else 0

    assert ap.install_unsloth(plan, venv, runner=runner) is True
    assert len(ran) == 1 and "--dry-run" in ran[0]          # resolved nothing, installed nothing
    assert ap.UNSLOTH_FLOOR in ran[0]                       # never accept a fossil unsloth + protobuf 3.x
    cpu_plan = _plan_for(monkeypatch, tmp_path)
    assert ap.install_unsloth(cpu_plan, venv, runner=lambda *a, **k: 1 / 0) is True   # never even tried


def test_existing_cpu_only_llama_cpp_cli_is_detected_from_cmake_cache(tmp_path) -> None:
    (tmp_path / "build").mkdir()
    assert ap.detect_llama_cli_backend(tmp_path) == ""
    (tmp_path / "build" / "CMakeCache.txt").write_text("GGML_CUDA:BOOL=OFF\nGGML_VULKAN:BOOL=OFF\n")
    assert ap.detect_llama_cli_backend(tmp_path) == "cpu"
    (tmp_path / "build" / "CMakeCache.txt").write_text("GGML_CUDA:BOOL=ON\n")
    assert ap.detect_llama_cli_backend(tmp_path) == "cuda"
    (tmp_path / "build" / ".fts-backend").write_text("hip\n")
    assert ap.detect_llama_cli_backend(tmp_path) == "hip"


def test_llama_cli_build_falls_back_through_backends_and_records_the_winner(monkeypatch, tmp_path) -> None:
    spec = nvidia([R3090], "580.178.04", "13.0")
    spec["which"] += ["glslc"]
    spec["files"]["/usr/include/vulkan/vulkan.h"] = "x"
    plan = _plan_for(monkeypatch, tmp_path, **spec)
    root = tmp_path / "llama.cpp"
    (root / "requirements").mkdir(parents=True)
    (root / "requirements" / "requirements-convert_hf_to_gguf.txt").write_text("")
    (root / "convert_hf_to_gguf.py").write_text("")
    monkeypatch.setattr(ap.shutil, "which", lambda c: "/usr/bin/" + c)
    cfg: list[str] = []

    def runner(cmd, env=None, shell_prefix=""):
        if cmd[0] == "cmake" and "-S" in cmd:
            cfg.append(next(a for a in cmd if a.startswith("-DGGML_")))
            if "-DGGML_CUDA=ON" in cmd:
                return 1                                     # CUDA configure fails
        if cmd[0] == "cmake" and "--build" in cmd:
            q = root / "build" / "bin"
            q.mkdir(parents=True, exist_ok=True)
            (q / "llama-quantize").write_text("")
        return 0

    assert ap.build_llama_cli(plan, root, tmp_path / "venv", runner=runner) is True
    assert cfg == ["-DGGML_CUDA=ON", "-DGGML_VULKAN=ON"]
    assert (root / "build" / ".fts-backend").read_text().strip() == "vulkan"


def test_cpu_only_cli_is_rebuilt_on_a_gpu_host(monkeypatch, tmp_path) -> None:
    plan = _plan_for(monkeypatch, tmp_path, **nvidia([R3090], "580.178.04", "13.0"))
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text("")
    (root / "build" / "bin" / "llama-quantize").write_text("")
    (root / "build" / "CMakeCache.txt").write_text("GGML_CUDA:BOOL=OFF\n")
    monkeypatch.setattr(ap.shutil, "which", lambda c: "/usr/bin/" + c)
    ran: list[list[str]] = []

    def runner(cmd, env=None, shell_prefix=""):
        ran.append(cmd)
        if "--build" in cmd:
            (root / "build" / "bin").mkdir(parents=True, exist_ok=True)
            (root / "build" / "bin" / "llama-quantize").write_text("")
        return 0

    assert ap.build_llama_cli(plan, root, tmp_path, only=True, runner=runner)
    assert any(c[0] == "cmake" and "-DGGML_CUDA=ON" in c for c in ran)
    assert (root / "build" / ".fts-backend").read_text().strip() == "cuda"


# ── CLI + the real shell installer on fake hardware ────────────────────────

def _bash_plan(tmp_path: Path, fixture: dict, *flags: str) -> dict:
    fx = tmp_path / "fx.json"
    fx.write_text(json.dumps({"system": "Linux", "machine": "x86_64", **fixture}))
    r = subprocess.run(["bash", str(ROOT / "install.sh"), "--plan", *flags], capture_output=True, text=True,
                       env={**os.environ, "FTS_ACCEL_FIXTURE": str(fx), "PATH": os.environ["PATH"],
                            "CUDA_VISIBLE_DEVICES": ""}, cwd=ROOT, check=False, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout[r.stdout.index("{"):])


def test_install_sh_plan_resolves_every_vendor(tmp_path: Path) -> None:
    assert _bash_plan(tmp_path, {})["torch_index"].endswith("/whl/cpu")
    a = _bash_plan(tmp_path, amd())
    assert (a["vendor"], a["torch_tag"], a["llama_backend"]) == ("amd", "rocm6.4", "hip")
    i = _bash_plan(tmp_path, {"which": ["sycl-ls"], "commands": {
        "sycl-ls": "[level_zero:gpu][level_zero:0] Intel(R) Arc(TM) B580 Graphics\n"}})
    assert (i["vendor"], i["torch_tag"]) == ("intel", "xpu")
    n = _bash_plan(tmp_path, nvidia([R3090], "580.1", "13.0"))
    assert (n["torch_tag"], n["llama_backend"]) == ("cu132", "cuda")
    # --cpu wins over detected hardware; --gpu overrides detection
    assert _bash_plan(tmp_path, nvidia([R3090], "580.1", "13.0"), "--cpu")["torch_tag"] == "cpu"
    assert _bash_plan(tmp_path, {}, "--gpu", "amd")["torch_tag"] == "rocm7.2"


def test_unsloth_profile_installs_without_the_stack_pins(monkeypatch, tmp_path) -> None:
    plan = _plan_for(monkeypatch, tmp_path, **nvidia([R3090], "580.1", "13.0"))
    venv = tmp_path / "v"
    venv.mkdir()
    ran: list[list[str]] = []
    assert ap.install_unsloth(plan, venv, profile=True, runner=lambda c, e=None, shell_prefix="": ran.append(c) or 0)
    assert len(ran) == 1 and "--dry-run" not in ran[0] and "stack-constraints.txt" not in " ".join(ran[0])


def test_source_builds_bypass_uvs_wheel_cache_so_cmake_flags_are_honored(monkeypatch, tmp_path) -> None:
    plan = _plan_for(monkeypatch, tmp_path, **nvidia([R3090], "580.178.04", "13.0"))
    src = next(a for a in ap.llama_py_attempts(plan, "py", []) if a[0].startswith("source build"))
    assert "--no-cache" in src[1] and "--no-binary" in src[1]
