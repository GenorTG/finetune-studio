"""Regression tests for the installer-diagnostics audit findings (subprocess/platform mocked)."""
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


def _cp(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ── 1. CPU-only host: compute_cap must never be an unbound local ────────────

def test_detect_on_cpu_only_host_without_any_gpu_tool() -> None:
    with patch("shutil.which", return_value=None), patch("pathlib.Path.exists", return_value=False):
        g = d.GpuInfo.detect()
    assert g.vendor == "none"
    assert g.compute_cap == ""


def test_detect_when_nvidia_smi_times_out_keeps_empty_compute_cap() -> None:
    def which(cmd: str) -> str | None:
        return "/usr/bin/nvidia-smi" if cmd == "nvidia-smi" else None

    with patch("shutil.which", side_effect=which), \
         patch("pathlib.Path.exists", return_value=False), \
         patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 10)):
        g = d.GpuInfo.detect()
    assert g.vendor == "none"
    assert g.compute_cap == ""


# ── 2. One probe must not share a single status across the six package checks ─

@pytest.fixture()
def fake_venv(tmp_path: Path) -> Path:
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True)
    py = venv / "bin" / "python"
    py.write_text("#!/bin/sh\n")
    py.chmod(0o755)
    return venv


def _inspect_with_probe(venv: Path, probe: subprocess.CompletedProcess) -> d.VenvInfo:
    with patch("install_diagnose._run") as run:
        run.side_effect = [
            _cp(stdout=json.dumps({"py": "3.13.0", "executable": "x"})),
            probe,
            _cp(returncode=1),  # torch
            _cp(returncode=1),  # torchaudio
            _cp(returncode=1), _cp(returncode=1), _cp(returncode=1), _cp(returncode=1),
        ]
        return d.inspect_venv(venv)


def test_single_missing_package_does_not_mark_the_other_five_missing(fake_venv: Path) -> None:
    info = _inspect_with_probe(fake_venv, _cp(returncode=1, stdout=json.dumps({"missing": ["datasets"]})))
    assert info.has_datasets is False
    assert info.has_fastapi and info.has_jinja2 and info.has_accelerate
    assert info.has_safetensors and info.has_huggingface_hub


def test_all_present_and_all_unknown_probe_outcomes(fake_venv: Path) -> None:
    ok = _inspect_with_probe(fake_venv, _cp(returncode=0))
    assert ok.has_fastapi and ok.has_datasets and ok.has_huggingface_hub
    # Probe crashed without a readable report: nothing can be claimed present.
    bad = _inspect_with_probe(fake_venv, _cp(returncode=1, stderr="segfault"))
    assert not (bad.has_fastapi or bad.has_datasets or bad.has_huggingface_hub)


def test_diagnose_names_only_the_actually_missing_package(fake_venv: Path, tmp_path: Path) -> None:
    info = _inspect_with_probe(fake_venv, _cp(returncode=1, stdout=json.dumps({"missing": ["datasets"]})))
    with patch.object(d, "inspect_venv", return_value=info), \
         patch.object(d.GpuInfo, "detect", return_value=d.GpuInfo("none", "(no GPU)", "", "", "", "")):
        issues = d.diagnose(fake_venv, tmp_path / "llama.cpp", check_service=False, force_cpu=True)
    deps = [i for i in issues if i.code == d.PIP_INSTALL_EDITABLE]
    assert len(deps) == 1
    assert deps[0].detail == "missing or broken: datasets"


# ── 3. repair() must not report success when a command failed ───────────────

CPU = d.GpuInfo("none", "(no GPU)", "", "", "", "")


def _repair(issues: list[d.Issue], tmp_path: Path, run_side_effect) -> tuple[bool, list[str], list[str]]:
    logs: list[str] = []
    with patch.object(d.GpuInfo, "detect", return_value=CPU), \
         patch("subprocess.run", side_effect=run_side_effect):
        ok, actions = d.repair(issues, tmp_path / ".venv", tmp_path / "llama.cpp", log=logs.append)
    return ok, actions, logs


def test_failed_cpu_torch_reinstall_is_not_reported_as_ok(tmp_path: Path) -> None:
    ok, actions, logs = _repair(
        [d.Issue(d.REINSTALL_TORCH, "x", "y")], tmp_path, lambda *a, **k: _cp(returncode=1, stderr="no wheel")
    )
    assert ok is False
    assert "torch CPU reinstall: FAIL" in actions
    assert any("no wheel" in line for line in logs)


def test_later_success_does_not_hide_earlier_failure(tmp_path: Path) -> None:
    results = iter([_cp(returncode=1, stderr="torch boom"), _cp(returncode=0)])
    ok, actions, _ = _repair(
        [d.Issue(d.REINSTALL_TORCH, "x", "y"), d.Issue(d.PIP_INSTALL_EDITABLE, "x", "y")],
        tmp_path, lambda *a, **k: next(results),
    )
    assert ok is False
    assert actions == ["torch CPU reinstall: FAIL", "pip -e .: ok"]


@pytest.mark.parametrize("code,label", [
    (d.PIP_INSTALL_EDITABLE, "pip -e ."),
    (d.REINSTALL_LLAMA_CPP, "llama-cpp reinstall"),
    (d.BUILD_LLAMA_CPP_CLI, "llama.cpp CLI build"),
    (d.INSTALL_SERVICE, "systemd service"),
])
def test_every_failed_step_flips_status(tmp_path: Path, code: str, label: str) -> None:
    ok, actions, _ = _repair([d.Issue(code, "x", "y")], tmp_path, lambda *a, **k: _cp(returncode=2, stderr="e"))
    assert ok is False
    assert f"{label}: FAIL" in actions


def test_command_that_cannot_run_is_a_recorded_failure_not_a_crash(tmp_path: Path) -> None:
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("pip", 600)

    ok, actions, _ = _repair([d.Issue(d.PIP_INSTALL_EDITABLE, "x", "y")], tmp_path, boom)
    assert ok is False
    assert actions[0].startswith("pip -e .: FAIL")


def test_all_commands_succeed_reports_ok(tmp_path: Path) -> None:
    ok, actions, _ = _repair([d.Issue(d.PIP_INSTALL_EDITABLE, "x", "y")], tmp_path, lambda *a, **k: _cp())
    assert ok is True
    assert actions == ["pip -e .: ok"]
