"""install_diagnose: GPU-stack failure modes + their `--repair` autofixes, on fake hardware.

Failure modes covered: GPU present but torch is a CPU build; torch built for the wrong
backend; llama-cpp-python without GPU offload; llama.cpp CLI built CPU-only / missing
llama-cli; GPU hardware without a driver (manual hint, never a silent CPU install);
missing bitsandbytes; Blackwell needing a source-built llama-cpp-python.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import install_diagnose as d

SMI = "nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv,noheader"


def cp(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def set_hw(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, vendor: str, **kw: object) -> None:
    spec: dict = {"system": "Linux", "machine": "x86_64"}
    if vendor == "nvidia":
        name, cc = kw.get("name", "NVIDIA GeForce RTX 3090"), kw.get("cc", "8.6")
        spec |= {"which": ["nvidia-smi", "nvcc"],
                 "commands": {SMI: f"0, {name}, {kw.get('driver', '580.178.04')}, {cc}\n",
                              "nvidia-smi": f"CUDA UMD Version: {kw.get('cuda', '13.0')}\n",
                              "nvcc --version": "Cuda compilation tools, release 13.4, V13.4.92\n",
                              "/usr/local/cuda/bin/nvcc --version": "Cuda compilation tools, release 13.4, V13.4.92\n"},
                 "files": {"/usr/local/cuda/bin/nvcc": "x", "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.13": "x"}}
    elif vendor == "amd":
        spec |= {"which": ["rocm-smi", "rocminfo"],
                 "files": {"/opt/rocm/.info/version": "6.4.1", "/dev/kfd": "", "/opt/rocm/bin/hipcc": "x"},
                 "commands": {"rocminfo": "Name: gfx1100\n", "rocm-smi --showproductname": "Card series: RX 7900\n"}}
    elif vendor == "intel":
        spec |= {"which": ["sycl-ls"], "commands": {"sycl-ls": "[level_zero:gpu][level_zero:0] Intel(R) Arc(TM) A770\n"}}
    elif vendor == "nvidia-nodriver":
        spec |= {"files": {"/sys/bus/pci/devices/0000:01:00.0/vendor": "0x10de",
                           "/sys/bus/pci/devices/0000:01:00.0/class": "0x030000"}}
    (tmp_path / "hw.json").write_text(json.dumps(spec))
    monkeypatch.setattr(d.ap, "_FIXTURE", None)


@pytest.fixture(autouse=True)
def hermetic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "hw.json").write_text(json.dumps({"system": "Linux", "machine": "x86_64"}))
    monkeypatch.setenv("FTS_ACCEL_FIXTURE", str(tmp_path / "hw.json"))
    monkeypatch.delenv("FTS_FORCE_VENDOR", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(d.ap, "_FIXTURE", None)
    monkeypatch.setattr(d, "probe_bitsandbytes", lambda py: "0.50.2")
    monkeypatch.setattr(d, "probe_llama_offload", lambda py: True)


def venv_info(tmp_path: Path, **over: object) -> d.VenvInfo:
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    base = {"path": venv, "python": venv / "bin" / "python", "py_version": "3.12.1", "torch_version": "2.14.1+cu132",
            "torch_has_cuda": True, "torchaudio_version": None, "torchaudio_importable": None,
            "transformers_version": "5.18.0", "llama_cpp_version": "0.3.36", "peft_version": "0.21", "trl_version": "1.14",
            "has_fastapi": True, "has_jinja2": True, "has_datasets": True, "has_accelerate": True,
            "has_safetensors": True, "has_huggingface_hub": True, "torch_kind": "cuda", "torch_xpu": False, "torch_mps": False}
    return d.VenvInfo(**{**base, **over})


def llama_dir(tmp_path: Path, backend: str | None = "cuda", cli: bool = True) -> Path:
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True, exist_ok=True)
    (root / "build" / "bin" / "llama-quantize").write_text("#!/bin/sh\n")
    (root / "build" / "bin" / "llama-quantize").chmod(0o755)
    if cli:
        (root / "build" / "bin" / "llama-cli").write_text("#!/bin/sh\n")
        (root / "build" / "bin" / "llama-cli").chmod(0o755)
    (root / "convert_hf_to_gguf.py").write_text("")
    if backend:
        (root / "build" / ".fts-backend").write_text(backend + "\n")
    return root


def run_diagnose(tmp_path: Path, info: d.VenvInfo, root: Path | None = None) -> list[d.Issue]:
    with patch.object(d, "inspect_venv", return_value=info):
        return d.diagnose(info.path, root or llama_dir(tmp_path), check_service=False)


def codes(issues: list[d.Issue]) -> set[str]:
    return {i.code for i in issues}


# ── diagnose: each failure mode ─────────────────────────────────────────────

def test_healthy_nvidia_stack_has_no_issues(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia")
    assert run_diagnose(tmp_path, venv_info(tmp_path)) == []


def test_gpu_present_but_torch_is_a_cpu_build(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia")
    issues = run_diagnose(tmp_path, venv_info(tmp_path, torch_version="2.14.1+cpu", torch_kind="cpu", torch_has_cuda=False))
    bad = [i for i in issues if i.code == d.REINSTALL_TORCH]
    assert bad and bad[0].severity == 2 and "CPU-only" in bad[0].detail
    assert "/whl/cu132" in bad[0].suggested_fix and "/whl/cpu" not in bad[0].suggested_fix


def test_cpu_torch_on_amd_and_intel_hosts_is_flagged_with_the_vendor_index(monkeypatch, tmp_path) -> None:
    for vendor, index in (("amd", "/whl/rocm6.4"), ("intel", "/whl/xpu")):
        set_hw(monkeypatch, tmp_path, vendor)
        issues = run_diagnose(tmp_path, venv_info(tmp_path, torch_version="2.14.1+cpu", torch_kind="cpu", torch_has_cuda=False))
        fix = next(i for i in issues if i.code == d.REINSTALL_TORCH).suggested_fix
        assert index in fix, (vendor, fix)


def test_wrong_backend_wheel_is_flagged(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia")
    issues = run_diagnose(tmp_path, venv_info(tmp_path, torch_version="2.14.1+rocm7.2", torch_kind="hip"))
    assert any(i.code == d.REINSTALL_TORCH and "wrong backend" in i.detail for i in issues)
    set_hw(monkeypatch, tmp_path, "amd")
    issues = run_diagnose(tmp_path, venv_info(tmp_path, torch_kind="cuda"))
    assert any(i.code == d.REINSTALL_TORCH and "needs a hip build" in i.detail for i in issues)


def test_rocm_and_xpu_builds_are_healthy_on_their_hardware(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "amd")
    assert d.REINSTALL_TORCH not in codes(run_diagnose(
        tmp_path, venv_info(tmp_path, torch_version="2.14.1+rocm6.4", torch_kind="hip")))
    set_hw(monkeypatch, tmp_path, "intel")
    assert d.REINSTALL_TORCH not in codes(run_diagnose(
        tmp_path, venv_info(tmp_path, torch_version="2.14.1+xpu", torch_kind="xpu", torch_has_cuda=False, torch_xpu=True)))


def test_cuda_build_that_cannot_see_the_gpu_is_flagged(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia")
    issues = run_diagnose(tmp_path, venv_info(tmp_path, torch_has_cuda=False))
    assert any(i.code == d.REINSTALL_TORCH and "cannot see the GPU" in i.detail for i in issues)


def test_gpu_build_torch_on_a_gpuless_host_is_only_a_warning(tmp_path) -> None:
    issues = run_diagnose(tmp_path, venv_info(tmp_path, torch_has_cuda=False))
    assert [i.severity for i in issues if i.code == d.REINSTALL_TORCH] == [1]


def test_absent_torchaudio_is_fine_but_a_broken_one_is_the_mixed_install_canary(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia")
    assert d.REINSTALL_TORCH not in codes(run_diagnose(tmp_path, venv_info(tmp_path, torchaudio_importable=None)))
    assert d.REINSTALL_TORCH in codes(run_diagnose(tmp_path, venv_info(tmp_path, torchaudio_importable=False)))


def test_llama_cpp_python_without_gpu_offload_is_flagged_on_gpu_hosts_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(d, "probe_llama_offload", lambda py: False)
    set_hw(monkeypatch, tmp_path, "nvidia")
    issues = run_diagnose(tmp_path, venv_info(tmp_path))
    bad = next(i for i in issues if i.code == d.REINSTALL_LLAMA_CPP)
    assert bad.severity == 2 and "NO GPU offload" in bad.detail and "GGML_CUDA" in bad.suggested_fix
    set_hw(monkeypatch, tmp_path, "none")
    assert d.REINSTALL_LLAMA_CPP not in codes(run_diagnose(tmp_path, venv_info(tmp_path, torch_has_cuda=False)))


def test_blackwell_needs_a_source_built_llama_cpp_python_until_the_marker_says_so(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia", name="NVIDIA GeForce RTX 5090", cc="12.0")
    info = venv_info(tmp_path)
    assert d.REINSTALL_LLAMA_CPP in codes(run_diagnose(tmp_path, info))
    (info.path / "fts-llama-cpp-python.json").write_text(json.dumps({"source": True, "arch": "120", "backend": "cuda"}))
    assert d.REINSTALL_LLAMA_CPP not in codes(run_diagnose(tmp_path, info))   # converges: no endless rebuild loop


def test_cpu_only_llama_cpp_cli_on_a_gpu_host_must_be_rebuilt(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia")
    issues = run_diagnose(tmp_path, venv_info(tmp_path), llama_dir(tmp_path, backend="cpu"))
    bad = next(i for i in issues if i.code == d.BUILD_LLAMA_CPP_CLI)
    assert "CPU-only" in bad.detail and "--rebuild-llama-cpp" in bad.suggested_fix and bad.severity == 2


def test_missing_llama_cli_binary_is_a_warning_missing_quantize_is_an_error(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia")
    issues = run_diagnose(tmp_path, venv_info(tmp_path), llama_dir(tmp_path, cli=False))
    assert [i.severity for i in issues if i.code == d.BUILD_LLAMA_CPP_CLI] == [1]
    root = llama_dir(tmp_path / "x")
    (root / "build" / "bin" / "llama-quantize").unlink()
    issues = run_diagnose(tmp_path, venv_info(tmp_path), root)
    assert [i.severity for i in issues if i.code == d.BUILD_LLAMA_CPP_CLI] == [2]


def test_gpu_without_driver_is_a_manual_issue_with_an_actionable_hint(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia-nodriver")
    issues = run_diagnose(tmp_path, venv_info(tmp_path, torch_has_cuda=False))
    drv = next(i for i in issues if i.code == d.INSTALL_GPU_DRIVER)
    assert "nvidia" in drv.detail.lower() and "driver" in drv.detail.lower() and drv.severity == 2
    assert not any(i.code == d.REINSTALL_TORCH and "cannot see the GPU" in i.detail for i in issues)


def test_missing_gpu_toolchain_is_reported_with_the_install_hint(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "amd")
    spec = json.loads((tmp_path / "hw.json").read_text())
    spec["files"].pop("/opt/rocm/bin/hipcc")
    (tmp_path / "hw.json").write_text(json.dumps(spec))
    monkeypatch.setattr(d.ap, "_FIXTURE", None)
    issues = run_diagnose(tmp_path, venv_info(tmp_path, torch_version="2.14.1+rocm6.4", torch_kind="hip"),
                          llama_dir(tmp_path, backend="cpu"))
    tool = next(i for i in issues if i.code == d.INSTALL_GPU_TOOLCHAIN)
    assert "hipcc" in tool.detail and "rocm-hip-sdk" in tool.suggested_fix


def test_missing_bitsandbytes_is_flagged_on_gpu_hosts_not_on_cpu_hosts(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(d, "probe_bitsandbytes", lambda py: None)
    set_hw(monkeypatch, tmp_path, "nvidia")
    assert d.INSTALL_BITSANDBYTES in codes(run_diagnose(tmp_path, venv_info(tmp_path)))
    set_hw(monkeypatch, tmp_path, "none")
    assert d.INSTALL_BITSANDBYTES not in codes(run_diagnose(tmp_path, venv_info(tmp_path, torch_has_cuda=False)))


# ── repair: each failure mode gets an autofix ───────────────────────────────

def repair(tmp_path: Path, issues: list[d.Issue], runs: list[tuple[list[str], dict | None]], rc_for=lambda cmd: 0):
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)

    def fake_run(cmd, **kw):
        runs.append((list(cmd), kw.get("env")))
        return cp(returncode=rc_for(list(cmd)))

    with patch("subprocess.run", side_effect=fake_run):
        return d.repair(issues, venv, tmp_path / "llama.cpp", log=lambda *a, **k: None)


def issue(code: str) -> d.Issue:
    return d.Issue(code, "x", "y", severity=2)


def test_torch_repair_uses_uv_with_the_gpu_index_and_verifies_the_backend(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia")
    runs: list = []
    ok, actions = repair(tmp_path, [issue(d.REINSTALL_TORCH)], runs)
    assert ok and actions == ["torch reinstall: ok (cu132)"]
    install = runs[0][0]
    assert install[:3] == ["uv", "pip", "install"] and "--python" in install     # uv venvs have no pip module
    assert install[install.index("--index-url") + 1].endswith("/whl/cu132")
    assert "torch" in install and "torchvision" in install and "torchaudio" not in install
    assert runs[1][0][1] == "-c" and "torch.version.cuda" in runs[1][0][2]        # backend verified


def test_gpu_torch_repair_steps_down_but_never_to_the_cpu_index(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia")
    runs: list = []
    ok, actions = repair(tmp_path, [issue(d.REINSTALL_TORCH)], runs, rc_for=lambda c: 1)
    assert ok is False and actions == ["torch reinstall: FAIL"]
    urls = [c[c.index("--index-url") + 1] for c, _ in runs if "--index-url" in c]
    assert urls[0].endswith("/cu132") and urls[1].endswith("/cu130")
    assert not any(u.endswith("/cpu") for u in urls)


def test_wrong_backend_wheel_triggers_the_next_index(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "amd")
    runs: list = []
    probes = {"n": 0}

    def rc(cmd):
        if cmd[1:2] == ["-c"]:
            probes["n"] += 1
            return 3 if probes["n"] == 1 else 0
        return 0

    ok, actions = repair(tmp_path, [issue(d.REINSTALL_TORCH)], runs, rc_for=rc)
    assert ok and actions == ["torch reinstall: ok (rocm6.3)"]


def test_cpu_torch_repair_only_when_no_gpu(tmp_path) -> None:
    runs: list = []
    ok, actions = repair(tmp_path, [issue(d.REINSTALL_TORCH)], runs)
    assert ok and actions == ["torch CPU reinstall: ok"]
    assert runs[0][0][runs[0][0].index("--index-url") + 1].endswith("/whl/cpu")


def test_llama_cpp_python_repair_prefers_wheel_verifies_offload_then_builds_from_source(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia")
    runs: list = []
    probes = {"n": 0}

    def rc(cmd):
        if cmd[1:2] == ["-c"]:
            probes["n"] += 1
            return 4 if probes["n"] == 1 else 0      # prebuilt wheel imports but has no GPU backend
        return 0

    ok, actions = repair(tmp_path, [issue(d.REINSTALL_LLAMA_CPP)], runs, rc_for=rc)
    assert ok and actions == ["llama-cpp reinstall: ok (source build (cuda))"]
    wheel, src = runs[0][0], runs[2]
    assert wheel[wheel.index("--extra-index-url") + 1].endswith("/whl/cu132")
    assert "--no-binary" in src[0] and "-DGGML_CUDA=ON" in src[1]["CMAKE_ARGS"]
    marker = json.loads((tmp_path / ".venv" / "fts-llama-cpp-python.json").read_text())
    assert marker["source"] is True and marker["backend"] == "cuda"


def test_blackwell_llama_repair_skips_the_prebuilt_wheel_and_targets_sm120(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia", name="NVIDIA GeForce RTX 5080", cc="12.0", driver="595.10", cuda="13.2")
    runs: list = []
    ok, actions = repair(tmp_path, [issue(d.REINSTALL_LLAMA_CPP)], runs)
    assert ok and actions == ["llama-cpp reinstall: ok (source build (cuda))"]
    cmd, env = runs[0]
    assert "--no-binary" in cmd and "--index-url" not in cmd
    assert "-DCMAKE_CUDA_ARCHITECTURES=120" in env["CMAKE_ARGS"]


def test_amd_and_intel_llama_cpp_python_repair_build_from_source_with_their_backend(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "amd")
    runs: list = []
    repair(tmp_path, [issue(d.REINSTALL_LLAMA_CPP)], runs)
    assert "-DGGML_HIP=ON" in runs[0][1]["CMAKE_ARGS"] and runs[0][1]["HIPCXX"].endswith("llvm/bin/clang")
    set_hw(monkeypatch, tmp_path, "intel")
    spec = json.loads((tmp_path / "hw.json").read_text())
    spec["files"] = {"/opt/intel/oneapi/compiler/latest/bin/icpx": "x"}
    (tmp_path / "hw.json").write_text(json.dumps(spec))
    monkeypatch.setattr(d.ap, "_FIXTURE", None)
    runs.clear()
    repair(tmp_path, [issue(d.REINSTALL_LLAMA_CPP)], runs)
    assert "-DGGML_SYCL=ON" in runs[0][1]["CMAKE_ARGS"]


def test_cli_repair_forces_a_gpu_rebuild_through_install_sh(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia")
    runs: list = []
    ok, actions = repair(tmp_path, [issue(d.BUILD_LLAMA_CPP_CLI)], runs)
    assert ok and actions == ["llama.cpp CLI build: ok"]
    assert runs[0][0][0] == "bash" and runs[0][0][2:] == ["--llama-cpp-only", "--rebuild-llama-cpp"]


def test_bitsandbytes_repair_installs_it_under_the_torch_pin(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia")
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "torch-constraints.txt").write_text("torch==2.14.1+cu132\n")
    runs: list = []
    ok, actions = repair(tmp_path, [issue(d.INSTALL_BITSANDBYTES)], runs)
    assert ok and actions == ["bitsandbytes: ok"]
    cmd = runs[0][0]
    assert cmd[-1].startswith("bitsandbytes") and "-c" in cmd


def test_manual_issues_are_reported_but_never_fail_or_run_sudo(monkeypatch, tmp_path) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia-nodriver")
    runs: list = []
    ok, actions = repair(tmp_path, [issue(d.INSTALL_GPU_DRIVER), issue(d.INSTALL_GPU_TOOLCHAIN),
                                    issue(d.INSTALL_CUDA_TOOLKIT)], runs)
    assert ok is True and runs == []
    assert all("MANUAL" in a for a in actions) and len(actions) == 3


def test_pip_editable_repair_syncs_the_parsers_extra_under_the_pin(monkeypatch, tmp_path) -> None:
    runs: list = []
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "torch-constraints.txt").write_text("torch==2.14.1\n")
    ok, actions = repair(tmp_path, [issue(d.PIP_INSTALL_EDITABLE)], runs)
    assert ok and actions == ["pip -e .: ok"]
    cmd = runs[0][0]
    assert cmd[:3] == ["uv", "pip", "install"] and "-c" in cmd and cmd[-1].endswith("[parsers]")


def test_cli_report_prints_gpu_and_plan_lines(monkeypatch, tmp_path, capsys) -> None:
    set_hw(monkeypatch, tmp_path, "nvidia")
    with patch.object(d, "inspect_venv", return_value=venv_info(tmp_path)):
        rc = d._main(["--venv", str(tmp_path / ".venv"), "--llama-cpp", str(llama_dir(tmp_path)),
                      "--no-service-check", "--check"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "plan:        torch=cu132 (cuda)  llama.cpp backend=cuda" in out and "(cuda)" in out
