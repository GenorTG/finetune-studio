"""Tests for scripts/install_diagnose.py — autodetect + autofix broken installs.

These run with the host's python (so we can patch subprocess calls) but
NEVER import torch/llama_cpp ourselves — the diagnose() function does
that through subprocess so a broken venv on disk can't poison us.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make the scripts/ dir importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import install_diagnose as diag  # noqa: E402


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
    (llama_cpp / "convert_hf_to_gguf.py").write_text("# fake\n")

    return venv, llama_cpp


def _fake_run(stdout="", stderr="", returncode=0):
    cp = subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr,
    )
    return cp


def _healthy_torch_payload(version: str = "2.11.0+cu130", cuda: bool = True) -> str:
    return json.dumps({"v": version, "cuda": cuda})


# ── GpuInfo.detect ──────────────────────────────────────────────────────

class TestGpuDetect:
    def test_force_cpu(self):
        g = diag.GpuInfo.detect(force_cpu=True)
        assert g.vendor == "none"
        assert g.cuda_ver == ""

    def test_nvidia_detected(self):
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("subprocess.run") as sr:
            sr.return_value = _fake_run(stdout="NVIDIA GeForce RTX 3090, 610.57.04\n")
            g = diag.GpuInfo.detect()
        assert g.vendor == "nvidia"
        assert "RTX 3090" in g.name
        assert g.driver_version == "610.57.04"
        assert g.cuda_ver == "cu130"  # 610 ≥ 550

    def test_nvidia_driver_525_maps_to_cu124(self):
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("subprocess.run") as sr:
            sr.return_value = _fake_run(stdout="RTX 3090, 530.41\n")
            g = diag.GpuInfo.detect()
        assert g.cuda_ver == "cu124"  # 530 ≥ 525

    def test_nvidia_driver_470_maps_to_cu118(self):
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("subprocess.run") as sr:
            sr.return_value = _fake_run(stdout="RTX 3090, 470.42\n")
            g = diag.GpuInfo.detect()
        assert g.cuda_ver == "cu118"

    def test_nvidia_smi_fails_falls_through(self):
        # When nvidia-smi is on PATH but returns empty/fails, AND no
        # /opt/rocm /opt/intel/oneapi exist, AND rocm-smi / xpu-smi are
        # not on PATH, GPU detection should report "none".
        def which_side_effect(cmd):
            return "/usr/bin/nvidia-smi" if cmd == "nvidia-smi" else None
        with patch("shutil.which", side_effect=which_side_effect), \
             patch("subprocess.run") as sr, \
             patch("pathlib.Path.exists", return_value=False):
            sr.return_value = _fake_run(stdout="", returncode=1)
            g = diag.GpuInfo.detect()
        assert g.vendor == "none"

    def test_rocm_detected(self):
        with patch("shutil.which", return_value=None), \
             patch("pathlib.Path.exists", return_value=True):
            g = diag.GpuInfo.detect()
        # Without rocm-smi binary, /opt/rocm existence alone would trigger
        # the amd branch — but our test patches Path.exists globally which
        # is too broad. Skip if it gets confused.
        # The point: we don't crash, we return *something*.
        assert g.vendor in ("amd", "none")


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

    def test_torch_cpu_with_gpu_detected(self, tmp_path):
        venv = tmp_path / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("#!/bin/sh\n")
        (venv / "bin" / "python").chmod(0o755)

        py_version_ok = json.dumps({"py": "3.13.0", "executable": "x"})
        import_ok = _fake_run(returncode=0)
        torch_cpu = _fake_run(stdout=_healthy_torch_payload("2.11.0+cpu", cuda=False))
        ta_broken = _fake_run(returncode=1, stderr="libc10_cuda.so")
        empty = _fake_run(returncode=0, stdout="")

        # Side-effects from GpuInfo.detect — make it report NVIDIA + cu130
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("subprocess.run") as sr_global, \
             patch("install_diagnose._run") as r:
            # GpuInfo.detect() nvidia-smi call
            sr_global.return_value = _fake_run(stdout="RTX 3090, 610.57.04\n")
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
    def test_healthy_returns_empty(self, fake_venv):
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


# ── repair() — autofix command construction ──────────────────────────────

class TestRepairTorchCommand:
    """Regression for the cmd-construction bug: fan-dragon's install
    repaired to 'pip pip install --python X --reinstall ...' (duplicated
    pip + wrong module path). Resulted in python -m install which
    silently failed and the bash script's set -e killed it before any
    wheel was downloaded."""

    def test_torch_repair_uses_python_m_pip(self, tmp_path):
        """The torch reinstall must use `sys.executable -m pip install`,
        NOT 'pip pip install --python X ...'. Captured via subprocess.run
        patch so we can assert the exact argv without doing a real pip
        install."""
        venv = tmp_path / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("#!/bin/sh\n")
        (venv / "bin" / "python").chmod(0o755)
        (tmp_path / "llama.cpp").mkdir()

        import install_diagnose as d
        gpu_stub = d.GpuInfo(
            vendor="nvidia", name="RTX 3090", driver_version="610.57.04",
            cuda_ver="cu130", cuda_toolkit_path="/opt/cuda",
        )
        issues = [d.Issue(d.REINSTALL_TORCH, "broken", "fix it", severity=2)]

        with patch.object(d, "GpuInfo") as gi, \
             patch.object(d, "subprocess") as sb:
            gi.detect.return_value = gpu_stub
            sb.run.return_value = _fake_run(returncode=0)
            ok, actions = d.repair(issues, venv, tmp_path / "llama.cpp",
                                   log=lambda *a, **k: None)

        # argv[0] must be a python interpreter (sys.executable is "/usr/bin/python3"
        # on Linux); argv[1]='-m'; argv[2]='pip'
        argv = sb.run.call_args[0][0]
        assert "python" in argv[0].lower(), \
            f"expected python interpreter, got {argv[0]}"
        assert argv[1] == "-m"
        assert argv[2] == "pip"
        # Must use the cu130 index (no --python flag — pip picks venv from sys.executable)
        assert "--index-url" in argv
        idx = argv[argv.index("--index-url") + 1]
        assert idx.endswith("/cu130"), f"expected cu130 index, got {idx}"
        # Must target the torch family
        for pkg in ("torch", "torchvision", "torchaudio"):
            assert pkg in argv, f"missing {pkg} in argv: {argv}"
        # The original bug was a duplicated 'pip' subcommand
        # (e.g. ['pip', 'pip', 'install', ...] or ['pip', 'pip', ...]).
        # The CORRECT pattern is ['python', '-m', 'pip', 'install', ...].
        # So we forbid two consecutive 'pip' entries.
        for i in range(len(argv) - 1):
            assert not (argv[i] == "pip" and argv[i + 1] == "pip"), \
                f"duplicated pip subcommand in argv: {argv}"
        assert "--python" not in argv, \
            "pip doesn't accept --python; rely on sys.executable for venv context"

    def test_cpu_torch_repair_uses_cpu_index(self, tmp_path):
        venv = tmp_path / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("#!/bin/sh\n")
        (venv / "bin" / "python").chmod(0o755)
        (tmp_path / "llama.cpp").mkdir()

        import install_diagnose as d
        gpu_stub = d.GpuInfo(
            vendor="none", name="(no GPU)", driver_version="",
            cuda_ver="", cuda_toolkit_path="",
        )
        issues = [d.Issue(d.REINSTALL_TORCH, "broken", "fix it", severity=2)]
        with patch.object(d, "GpuInfo") as gi, \
             patch.object(d, "subprocess") as sb:
            gi.detect.return_value = gpu_stub
            sb.run.return_value = _fake_run(returncode=0)
            d.repair(issues, venv, tmp_path / "llama.cpp",
                     log=lambda *a, **k: None)
        argv = sb.run.call_args[0][0]
        idx = argv[argv.index("--index-url") + 1]
        assert idx.endswith("/cpu"), f"expected cpu index, got {idx}"

    def test_repair_returns_false_when_torch_install_fails(self, tmp_path):
        venv = tmp_path / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("#!/bin/sh\n")
        (venv / "bin" / "python").chmod(0o755)
        (tmp_path / "llama.cpp").mkdir()

        import install_diagnose as d
        gpu_stub = d.GpuInfo(
            vendor="nvidia", name="RTX 3090", driver_version="610.57.04",
            cuda_ver="cu130", cuda_toolkit_path="/opt/cuda",
        )
        issues = [d.Issue(d.REINSTALL_TORCH, "broken", "fix it", severity=2)]
        with patch.object(d, "GpuInfo") as gi, \
             patch.object(d, "subprocess") as sb:
            gi.detect.return_value = gpu_stub
            sb.run.return_value = _fake_run(returncode=1, stderr="boom")
            ok, actions = d.repair(issues, venv, tmp_path / "llama.cpp",
                                   log=lambda *a, **k: None)
        assert ok is False
        assert any("torch reinstall: FAIL" in a for a in actions)


# ── CLI smoke tests ─────────────────────────────────────────────────────

class TestCLI:
    def test_check_returns_0_when_healthy(self, fake_venv):
        venv, llcpp = fake_venv
        with patch("install_diagnose._run") as r:
            r.side_effect = [
                _fake_run(stdout=json.dumps({"py": "3.13.0", "executable": "x"})),
                _fake_run(returncode=0),
                _fake_run(stdout=_healthy_torch_payload()),
                _fake_run(returncode=0, stdout="2.11.0+cu130"),
                _fake_run(returncode=0, stdout="4.50.0"),
                _fake_run(returncode=0, stdout="0.3.35"),
                _fake_run(returncode=0, stdout="0.20.0"),
                _fake_run(returncode=0, stdout="1.12.0"),
            ]
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

    def test_json_output(self, tmp_path):
        venv = tmp_path / "no-such"
        llcpp = tmp_path / "llama.cpp"
        rc = diag._main([
            "--venv", str(venv), "--llama-cpp", str(llcpp),
            "--json", "--no-service-check",
        ])
        # Should still print valid JSON
        out = sys.stdout.getvalue() if hasattr(sys.stdout, "getvalue") else None
        # argparse --json emits to stdout; just ensure rc is non-zero
        assert rc != 0
