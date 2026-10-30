"""CLI parity for Settings → Compute device."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from finetune_studio.accel import env, saved_choice
from finetune_studio.cli import _registry
from finetune_studio.webui import compute_device as device_settings

DEVICE_ID = "nvidia:GPU-test-3090"


@pytest.fixture
def fake_devices(monkeypatch: pytest.MonkeyPatch) -> list[device_settings.DeviceInfo]:
    devices = [device_settings.DeviceInfo(
        id=DEVICE_ID, index=0, name="NVIDIA GeForce RTX 3090", vendor="NVIDIA", runtime="CUDA 13.0",
        total_gb=24.0, free_gb=18.5, visible=True, active=True,
    )]
    monkeypatch.setattr(device_settings, "list_devices", lambda: devices)
    monkeypatch.setattr(device_settings, "restart_command", lambda: "fts restart web")
    monkeypatch.setattr(env, "_applied", env.AppliedPolicy("default", saved_choice.SavedChoice()))
    return devices


def run_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
            *argv: str) -> tuple[int, str, str]:
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["fts", *argv])
    try:
        _registry.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    else:
        code = 0
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_show_json_matches_shared_report(monkeypatch, capsys, fake_devices, tmp_path: Path) -> None:
    monkeypatch.setenv("FTS_ROOT", str(tmp_path))
    saved_choice.save(saved_choice.SavedChoice(mode="cpu"))
    expected = device_settings.report(saved_choice.load(), fake_devices)

    code, out, err = run_cli(monkeypatch, capsys, "compute-device", "show", "--json")

    assert code == 0 and not err
    assert json.loads(out) == expected
    assert json.loads(out)["saved"]["mode"] == "cpu"


def test_show_text_explains_device_and_restart(monkeypatch, capsys, fake_devices, tmp_path: Path) -> None:
    monkeypatch.setenv("FTS_ROOT", str(tmp_path))
    saved_choice.save(saved_choice.SavedChoice(mode="gpu", vendor="nvidia", uuid="GPU-test-3090",
                                               name="NVIDIA GeForce RTX 3090", index=0))
    # Current process was started with Auto, so the saved GPU needs a restart.
    code, out, err = run_cli(monkeypatch, capsys, "compute-device", "show")

    assert code == 0 and not err
    assert "Saved choice: NVIDIA GeForce RTX 3090" in out
    assert f"device id: {DEVICE_ID} · available" in out
    assert "Restart required to apply: fts restart web" in out


def test_set_gpu_uses_shared_validation_and_persists_only_under_fts_root(
    monkeypatch, capsys, fake_devices, tmp_path: Path,
) -> None:
    root = tmp_path / "fts"
    monkeypatch.setenv("FTS_ROOT", str(root))

    code, out, err = run_cli(monkeypatch, capsys, "compute-device", "set", "gpu", "--device-id", DEVICE_ID, "--json")

    assert code == 0 and not err
    report = json.loads(out)
    assert report["saved"] == {
        "mode": "gpu", "device_id": DEVICE_ID, "name": "NVIDIA GeForce RTX 3090", "available": True,
    }
    assert report["restart_required"] is True
    assert saved_choice.load() == saved_choice.SavedChoice(
        mode="gpu", vendor="nvidia", uuid="GPU-test-3090", name="NVIDIA GeForce RTX 3090", index=0,
    )
    assert saved_choice.choice_path().is_relative_to(root)


def test_invalid_gpu_is_rejected_without_persisting(monkeypatch, capsys, fake_devices, tmp_path: Path) -> None:
    root = tmp_path / "fts"
    monkeypatch.setenv("FTS_ROOT", str(root))

    code, out, err = run_cli(monkeypatch, capsys, "compute-device", "set", "gpu", "--device-id", "nvidia:missing")

    assert code == 1 and not out
    assert "unknown device" in err and DEVICE_ID in err and "Traceback" not in err
    assert not saved_choice.choice_path().exists()


def test_set_auto_rejects_device_id(monkeypatch, capsys, fake_devices, tmp_path: Path) -> None:
    monkeypatch.setenv("FTS_ROOT", str(tmp_path))

    code, out, err = run_cli(monkeypatch, capsys, "compute-device", "set", "auto", "--device-id", DEVICE_ID)

    assert code == 1 and not out
    assert "only valid with mode 'gpu'" in err
    assert not saved_choice.choice_path().exists()


def test_save_error_is_user_facing(monkeypatch, capsys, fake_devices) -> None:
    def fail(_choice):
        raise PermissionError("read-only device settings")

    monkeypatch.setattr(saved_choice, "save", fail)
    code, out, err = run_cli(monkeypatch, capsys, "compute-device", "set", "cpu")

    assert code == 1 and not out
    assert "read-only device settings" in err and "Traceback" not in err
