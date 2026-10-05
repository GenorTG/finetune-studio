"""Tests for scripts/install_diagnose.py — autodetect + autofix broken installs.

These run with the host's python (so we can patch subprocess calls) but
NEVER import torch/llama_cpp ourselves — the diagnose() function does
that through subprocess so a broken venv on disk can't poison us.
"""
from __future__ import annotations

import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the scripts/ dir importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import install_diagnose as diag  # noqa: I001


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def fake_venv(tmp_path):
    """Create a fake venv dir with a working python that returns
    whatever stdout the caller mocked. Default behaviour: venv exists,
    python works, all key imports succeed, torch has CUDA, torchaudio
    imports, llama.cpp CLI is built."""
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True)
    py = venv / "bin" / "python"
    py.write_text("#!/usr/bin/env python3\nprint('fake py')\n")
    py.chmod(0o755)

    llama_cpp = tmp_path / "llama.cpp"
    (llama_cpp / "build/bin").mkdir(parents=True)
    (llama_cpp / "build/bin/llama-quantize").write_text("#!/bin/sh\n")
    (llama_cpp / "build/bin/llama-quantize").chmod(0o755)
    (llama_cpp / "build/bin/llama-cli").write_text("#!/bin/sh\n")
    (llama_cpp / "build/bin/llama-cli").chmod(0o755)
    (llama_cpp / "convert_hf_to_gguf.py").write_text("# fake\n")

    return venv, llama_cpp


def _fake_run(stdout="", stderr="", returncode=0):
    cp = subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr,
    )
    return cp


def _healthy_torch_payload(version: str = "2.11.0+cu130", cuda: bool = True) -> str:
    return json.dumps({"v": version, "cuda": cuda})


@pytest.fixture(autouse=True)
def hermetic_host(monkeypatch, tmp_path):
    """Never read the real machine: no GPU hardware, healthy bitsandbytes + llama offload probes.
    Tests that need hardware override with `fake_hw` (see tests/test_accel_plan.py fixtures)."""
    hw = tmp_path / "hw.json"
    hw.write_text(json.dumps({"system": "Linux", "machine": "x86_64"}))
    monkeypatch.setenv("FTS_ACCEL_FIXTURE", str(hw))
    monkeypatch.delenv("FTS_FORCE_VENDOR", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(diag.ap, "_FIXTURE", None)
    monkeypatch.setattr(diag, "probe_bitsandbytes", lambda py: "0.50.2")
    monkeypatch.setattr(diag, "probe_llama_offload", lambda py: True)


def _nvidia_hw(monkeypatch, tmp_path, name="NVIDIA GeForce RTX 3090", driver="580.178.04", cc="8.6", cuda="13.0"):
    smi = "nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv,noheader"
    spec = {"system": "Linux", "machine": "x86_64", "which": ["nvidia-smi", "nvcc"],
            "commands": {smi: f"0, {name}, {driver}, {cc}\n",
                         "nvidia-smi": f"CUDA UMD Version: {cuda}\n",
                         "nvcc --version": "Cuda compilation tools, release 13.4, V13.4.92\n",
                              "/usr/local/cuda/bin/nvcc --version": "Cuda compilation tools, release 13.4, V13.4.92\n"},
            "files": {"/usr/local/cuda/bin/nvcc": "x"}}
    (tmp_path / "hw.json").write_text(json.dumps(spec))
    monkeypatch.setattr(diag.ap, "_FIXTURE", None)


# ── GpuInfo.detect ──────────────────────────────────────────────────────

class TestGpuDetect:
    def test_force_cpu(self):
        g = diag.GpuInfo.detect(force_cpu=True)
        assert g.vendor == "none"
        assert g.cuda_ver == ""

    def test_nvidia_detected(self, monkeypatch, tmp_path):
        _nvidia_hw(monkeypatch, tmp_path, driver="610.57.04", cuda="13.2")
        g = diag.GpuInfo.detect()
        assert g.vendor == "nvidia"
        assert "RTX 3090" in g.name
        assert g.driver_version == "610.57.04"
        assert g.cuda_ver == "cu132"

    @pytest.mark.parametrize("driver,cuda,tag", [
        ("580.178.04", "13.0", "cu132"),   # CUDA-13 driver runs any cu13x wheel (verified live)
        ("550.54", "12.4", "cu124"),
        ("530.41", "12.1", "cu121"),
        ("470.42", "11.4", "cu118"),
    ])
    def test_nvidia_driver_maps_to_newest_supported_index(self, monkeypatch, tmp_path, driver, cuda, tag):
        _nvidia_hw(monkeypatch, tmp_path, driver=driver, cuda=cuda)
        assert diag.GpuInfo.detect().cuda_ver == tag

    def test_nvidia_smi_fails_falls_through(self, monkeypatch, tmp_path):
        spec = {"system": "Linux", "machine": "x86_64", "which": ["nvidia-smi"],
                "commands": {"nvidia-smi": {"rc": 9, "out": "NVIDIA-SMI has failed"}}}
        (tmp_path / "hw.json").write_text(json.dumps(spec))
        monkeypatch.setattr(diag.ap, "_FIXTURE", None)
        assert diag.GpuInfo.detect().vendor == "none"

    def test_rocm_detected(self, monkeypatch, tmp_path):
        spec = {"system": "Linux", "machine": "x86_64", "which": ["rocm-smi"], "files": {"/opt/rocm/.info/version": "6.4.1"}}
        (tmp_path / "hw.json").write_text(json.dumps(spec))
        monkeypatch.setattr(diag.ap, "_FIXTURE", None)
        assert diag.GpuInfo.detect().vendor == "amd"


# ── diagnose() — broken venv cases ──────────────────────────────────────

class TestDiagnoseBrokenVenv:
    def test_missing_venv_returns_recreate(self, tmp_path):
        venv = tmp_path / "no-such-venv"
        llcpp = tmp_path / "llama.cpp"
        issues = diag.diagnose(venv, llcpp, check_service=False)
        assert any(i.code == diag.RECREATE_VENV for i in issues)

    def test_venv_python_broken_returns_recreate(self, tmp_path):
        # venv dir exists, python "exists" but can't import sys
        venv = tmp_path / ".venv"
        (venv / "bin").mkdir(parents=True)
        py = venv / "bin" / "python"
        py.write_text("#!/usr/bin/env python3\n")
        py.chmod(0o755)
        with patch("install_diagnose._run") as r:
            r.return_value = _fake_run(returncode=1, stderr="boom")
            issues = diag.diagnose(venv, tmp_path / "llama.cpp",
                                   check_service=False)
        assert any(i.code == diag.RECREATE_VENV and i.severity >= 3
                   for i in issues)


# ── diagnose() — mixed torch install ────────────────────────────────────

class TestDiagnoseMixedTorch:
    """The exact bug we hit on fan-dragon: torch+cpu with torchaudio+cu130
    and torchaudio fails to import because libc10_cuda.so is missing."""

    def test_torch_cpu_with_gpu_detected(self, tmp_path, monkeypatch):
        venv = tmp_path / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("#!/bin/sh\n")
        (venv / "bin" / "python").chmod(0o755)

        py_version_ok = json.dumps({"py": "3.13.0", "executable": "x"})
        import_ok = _fake_run(returncode=0)
        torch_cpu = _fake_run(stdout=_healthy_torch_payload("2.11.0+cpu", cuda=False))
        ta_broken = _fake_run(returncode=1, stderr="libc10_cuda.so")
        empty = _fake_run(returncode=0, stdout="")

        _nvidia_hw(monkeypatch, tmp_path, driver="610.57.04", cuda="13.2")
        with patch("install_diagnose._run") as r:
            # _run() called by inspect_venv (multiple)
            r.side_effect = [
                _fake_run(stdout=py_version_ok),   # basic sys check
                import_ok,                          # fastapi/jinja2 check
                torch_cpu,                          # torch check (cpu)
                ta_broken,                          # torchaudio check (fails)
                empty,                              # transformers (no output ok)
                empty,                              # llama_cpp
                empty,                              # peft
                empty,                              # trl
            ]
            issues = diag.diagnose(venv, tmp_path / "llama.cpp",
                                   check_service=False)

        codes = {i.code for i in issues}
        assert diag.REINSTALL_TORCH in codes, issues
        # The issue should mention the GPU info to help the user
        msg = " ".join(i.detail for i in issues if i.code == diag.REINSTALL_TORCH)
        assert "CPU-only" in msg or "+cpu" in msg
        assert any(i.severity == 2 for i in issues if i.code == diag.REINSTALL_TORCH)

    def test_torchaudio_broken_triggers_reinstall_even_if_torch_ok(self, tmp_path):
        """The torchaudio+libc10_cuda.so failure is the canary for mixed
        installs — it must be flagged even if torch reports CUDA."""
        venv = tmp_path / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("#!/bin/sh\n")
        (venv / "bin" / "python").chmod(0o755)

        py_version_ok = json.dumps({"py": "3.13.0", "executable": "x"})
        import_ok = _fake_run(returncode=0)
        torch_cu = _fake_run(stdout=_healthy_torch_payload("2.11.0+cu130", cuda=True))
        ta_broken = _fake_run(returncode=1, stderr="Could not load libc10_cuda.so")
        empty = _fake_run(returncode=0, stdout="")

        with patch("install_diagnose._run") as r:
            r.side_effect = [
                _fake_run(stdout=py_version_ok),
                import_ok,
                torch_cu,
                ta_broken,
                empty, empty, empty, empty,
            ]
            issues = diag.diagnose(venv, tmp_path / "llama.cpp",
                                   check_service=False)

        # Even though torch reports CUDA, torchaudio failure is a critical
        # mixed-install symptom
        assert any(i.code == diag.REINSTALL_TORCH for i in issues)
        msg = " ".join(i.detail for i in issues if i.code == diag.REINSTALL_TORCH)
        assert "libc10_cuda" in msg or "mixed" in msg.lower()


# ── diagnose() — missing deps / llama.cpp CLI ───────────────────────────

class TestDiagnoseMissingParts:
    def test_missing_fastapi_flagged(self, tmp_path):
        venv = tmp_path / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("#!/bin/sh\n")
        (venv / "bin" / "python").chmod(0o755)

        with patch("install_diagnose._run") as r:
            r.side_effect = [
                _fake_run(stdout=json.dumps({"py": "3.13.0", "executable": "x"})),
                _fake_run(returncode=1, stderr="ModuleNotFoundError: fastapi"),
                _fake_run(returncode=1),    # torch
                _fake_run(returncode=1),    # torchaudio (will be flagged)
                _fake_run(returncode=1),    # transformers
                _fake_run(returncode=1),    # llama_cpp
                _fake_run(returncode=1),    # peft
                _fake_run(returncode=1),    # trl
            ]
            issues = diag.diagnose(venv, tmp_path / "llama.cpp",
                                   check_service=False)
        assert any(i.code == diag.PIP_INSTALL_EDITABLE for i in issues)

    def test_llama_cpp_cli_missing(self, fake_venv, tmp_path):
        venv, llcpp = fake_venv
        # Delete the fake llama.cpp CLI
        shutil.rmtree(llcpp / "build")
        with patch("install_diagnose._run") as r:
            r.side_effect = [
                _fake_run(stdout=json.dumps({"py": "3.13.0", "executable": "x"})),
                _fake_run(returncode=0),  # fastapi etc
                _fake_run(stdout=_healthy_torch_payload()),  # torch
                _fake_run(returncode=0, stdout="2.11.0+cu130"),  # torchaudio
                _fake_run(returncode=0, stdout="4.50.0"),  # transformers
                _fake_run(returncode=0, stdout="0.3.35"),  # llama_cpp
                _fake_run(returncode=0, stdout="0.20.0"),  # peft
                _fake_run(returncode=0, stdout="1.12.0"),  # trl
            ]
            issues = diag.diagnose(venv, llcpp, check_service=False)
        assert any(i.code == diag.BUILD_LLAMA_CPP_CLI for i in issues)


# ── diagnose() — healthy ────────────────────────────────────────────────

class TestDiagnoseHealthy:
    def test_healthy_returns_empty(self, fake_venv, monkeypatch, tmp_path):
        _nvidia_hw(monkeypatch, tmp_path)
        venv, llcpp = fake_venv
        with patch("install_diagnose._run") as r:
            r.side_effect = [
                _fake_run(stdout=json.dumps({"py": "3.13.0", "executable": "x"})),
                _fake_run(returncode=0),  # fastapi etc
                _fake_run(stdout=_healthy_torch_payload()),  # torch
                _fake_run(returncode=0, stdout="2.11.0+cu130"),  # torchaudio
                _fake_run(returncode=0, stdout="4.50.0"),  # transformers
                _fake_run(returncode=0, stdout="0.3.35"),  # llama_cpp
                _fake_run(returncode=0, stdout="0.20.0"),  # peft
                _fake_run(returncode=0, stdout="1.12.0"),  # trl
            ]
            issues = diag.diagnose(venv, llcpp, check_service=False,
                                   force_cpu=False)
        # On the test host, no GPU is detected → no GPU-vs-CUDA mismatches
        codes = [i.code for i in issues]
        assert diag.RECREATE_VENV not in codes
        assert diag.REINSTALL_TORCH not in codes
        assert diag.REINSTALL_LLAMA_CPP not in codes
        assert diag.BUILD_LLAMA_CPP_CLI not in codes


# ── CLI smoke tests ─────────────────────────────────────────────────────

class TestCLI:
    def test_check_returns_0_when_healthy(self, fake_venv, monkeypatch, tmp_path):
        _nvidia_hw(monkeypatch, tmp_path)
        venv, llcpp = fake_venv
        with patch("install_diagnose._run") as r:
            # _main inspects the venv twice (diagnose + summary print), so cycle
            # the 8 probe responses instead of exhausting a one-shot list.
            r.side_effect = itertools.cycle([
                _fake_run(stdout=json.dumps({"py": "3.13.0", "executable": "x"})),
                _fake_run(returncode=0),
                _fake_run(stdout=_healthy_torch_payload()),
                _fake_run(returncode=0, stdout="2.11.0+cu130"),
                _fake_run(returncode=0, stdout="4.50.0"),
                _fake_run(returncode=0, stdout="0.3.35"),
                _fake_run(returncode=0, stdout="0.20.0"),
                _fake_run(returncode=0, stdout="1.12.0"),
            ])
            rc = diag._main([
                "--venv", str(venv), "--llama-cpp", str(llcpp),
                "--check", "--no-service-check",
            ])
        assert rc == 0

    def test_check_returns_2_when_critical(self, tmp_path):
        # No venv at all
        venv = tmp_path / "no-such"
        llcpp = tmp_path / "llama.cpp"
        rc = diag._main([
            "--venv", str(venv), "--llama-cpp", str(llcpp),
            "--check", "--no-service-check",
        ])
        assert rc >= 2

    def test_json_output(self, tmp_path, capsys):
        venv = tmp_path / "no-such"
        llcpp = tmp_path / "llama.cpp"
        rc = diag._main([
            "--venv", str(venv), "--llama-cpp", str(llcpp),
            "--json", "--no-service-check",
        ])
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert isinstance(payload, list)
        assert payload
        assert all({"code", "severity", "detail", "suggested_fix"} <= issue.keys()
                   for issue in payload)
        assert rc != 0
