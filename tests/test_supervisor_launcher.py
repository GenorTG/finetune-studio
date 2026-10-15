"""Launcher detection must not be fooled by an inherited INVOCATION_ID."""
from __future__ import annotations

from finetune_studio.supervisor.launcher import detect


def test_unit_cgroup_means_systemd(tmp_path):
    f = tmp_path / "cgroup"
    f.write_text("0::/user.slice/user-1000.slice/user@1000.service/app.slice/finetune-studio.service\n")
    assert detect(f) == "systemd"


def test_other_service_cgroup_with_inherited_env_is_manual(tmp_path, monkeypatch):
    monkeypatch.setenv("INVOCATION_ID", "abc123")  # leaked from the shell's own unit
    f = tmp_path / "cgroup"
    f.write_text("0::/user.slice/user-1000.slice/user@1000.service/app.slice/openclaw-gateway.service\n")
    assert detect(f) == "manual"


def test_unreadable_cgroup_is_manual(tmp_path):
    assert detect(tmp_path / "missing") == "manual"
