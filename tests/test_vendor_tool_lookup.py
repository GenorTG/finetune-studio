"""Vendor CLIs are found off PATH too: a systemd --user unit gets the system PATH only,
genorbox1's /usr/bin/nvidia-smi is a dangling alternatives link and the real binary lives
in ~/.local/bin, so the Compute device card listed no GPU on the live host (2026-10-10)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from finetune_studio.accel import env as accel_env
from finetune_studio.webui import gpu_probe


def _fake_tool(d: Path, name: str, output: str) -> Path:
    tool = d / name
    tool.write_text(f"#!/bin/sh\nprintf '%s' '{output}'\n")
    tool.chmod(0o755)
    return tool


@pytest.fixture
def off_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty PATH plus one install dir the resolver knows about."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    (tmp_path / "empty-bin").mkdir()
    install = tmp_path / "local-bin"
    install.mkdir()
    monkeypatch.setattr(accel_env, "_TOOL_DIRS", (str(install),))
    return install


def test_path_hit_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    on_path = tmp_path / "bin"
    on_path.mkdir()
    tool = _fake_tool(on_path, "nvidia-smi", "x")
    monkeypatch.setenv("PATH", str(on_path))
    assert accel_env.vendor_tool("nvidia-smi") == str(tool)


def test_install_dir_fallback(off_path: Path) -> None:
    tool = _fake_tool(off_path, "nvidia-smi", "x")
    assert accel_env.vendor_tool("nvidia-smi") == str(tool)


def test_dangling_symlink_is_not_a_tool(off_path: Path) -> None:
    os.symlink(off_path / "gone", off_path / "nvidia-smi")
    assert accel_env.vendor_tool("nvidia-smi") is None


def test_missing_tool_is_none(off_path: Path) -> None:
    assert accel_env.vendor_tool("rocm-smi") is None


def test_accel_list_nvidia_uses_fallback_dir(off_path: Path) -> None:
    _fake_tool(off_path, "nvidia-smi", "0, GPU-abc, NVIDIA GeForce RTX 3090")
    gpus = accel_env.list_nvidia()
    assert [(g.index, g.ident, g.name) for g in gpus] == [(0, "GPU-abc", "NVIDIA GeForce RTX 3090")]


def test_webui_probe_uses_fallback_dir(off_path: Path) -> None:
    _fake_tool(off_path, "nvidia-smi", "hello")
    assert gpu_probe._run(["nvidia-smi", "-L"]) == "hello"


def test_webui_probe_missing_tool_is_none(off_path: Path) -> None:
    assert gpu_probe._run(["nvidia-smi", "-L"]) is None
