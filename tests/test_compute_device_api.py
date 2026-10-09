"""``/api/system/compute-device``: devices, saved vs effective choice, restart_required, env override.

CPU-only: the GPU inventory and the accelerator are faked; the saved file lives under the per-test
FTS_ROOT that conftest's ``temp_db`` provides.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from finetune_studio.accel import device as accel_device
from finetune_studio.accel import env, saved_choice
from finetune_studio.accel.saved_choice import SavedChoice
from finetune_studio.webui import compute_device as cd

ROWS = [
    {"index": 0, "ident": "GPU-297ed2f9-b4d6", "name": "NVIDIA GeForce RTX 3090", "vendor": "nvidia",
     "total_gb": 24.0, "free_gb": 23.1},
    {"index": 1, "ident": "GPU-b6f0e3aa-5f4b", "name": "NVIDIA GeForce GTX 1070", "vendor": "nvidia",
     "total_gb": 8.0, "free_gb": 7.9},
]
ID_3090, ID_1070 = "nvidia:GPU-297ed2f9-b4d6", "nvidia:GPU-b6f0e3aa-5f4b"
URL = "/api/system/compute-device"


def _acc(kind: str = "cuda", name: str = "NVIDIA GeForce RTX 3090") -> accel_device.Accelerator:
    gpu = kind != "cpu"
    return accel_device.Accelerator(
        kind=kind, index=0, name=name, total_gb=24.0 if gpu else 0.0, free_gb=23.0 if gpu else 0.0,
        compute_capability=(8, 6) if gpu else (0, 0), supports_bf16=gpu, supports_flash_attention=gpu,
        supports_4bit=gpu, runtime="CUDA 13.0" if gpu else "CPU")


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch):
    """Two-GPU host; returns a setter for the accelerator and the start-up policy this process ran."""
    monkeypatch.setattr(cd, "_nvidia_rows", lambda: [dict(r) for r in ROWS])
    monkeypatch.setattr(cd, "_amd_rows", list)
    for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        monkeypatch.setenv(var, "x")
        monkeypatch.delenv(var)
    monkeypatch.setattr(accel_device, "_cached", _acc())

    def start(applied: env.AppliedPolicy | None = None, acc: accel_device.Accelerator | None = None) -> None:
        monkeypatch.setattr(env, "_applied", applied or env.AppliedPolicy("default", SavedChoice()))
        if acc is not None:
            monkeypatch.setattr(accel_device, "_cached", acc)

    start()
    yield start
    accel_device.reset_cache()


# ── GET ──────────────────────────────────────────────────────────────────────
def test_get_reports_devices_and_defaults(client, host) -> None:
    d = client.get(URL).json()
    assert [(g["id"], g["index"], g["total_gb"], g["free_gb"]) for g in d["devices"]] == [
        (ID_3090, 0, 24.0, 23.1), (ID_1070, 1, 8.0, 7.9)]
    assert [g["active"] for g in d["devices"]] == [True, False]
    assert d["devices"][0]["runtime"] == "CUDA 13.0" and d["devices"][1]["vendor"] == "NVIDIA"
    assert d["saved"] == {"mode": "auto", "device_id": "", "name": "", "available": True}
    assert d["effective"]["device"] == "cuda:0" and d["effective"]["device_id"] == ID_3090
    assert d["effective"]["source"] == "default" and d["effective"]["is_gpu"] is True
    assert d["restart_required"] is False and d["overridden_by_env"] is False and d["env_overrides"] == []
    assert d["restart_command"] == "systemctl --user restart finetune-studio"


def test_masked_card_is_listed_but_not_visible(client, host, monkeypatch: pytest.MonkeyPatch) -> None:
    """A card hidden from this process can still be chosen (it is the selector's whole point)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-297ed2f9-b4d6")
    d = client.get(URL).json()
    assert [(g["id"], g["visible"]) for g in d["devices"]] == [(ID_3090, True), (ID_1070, False)]
    assert d["effective"]["visibility"] == {"CUDA_VISIBLE_DEVICES": "GPU-297ed2f9-b4d6"}


def test_visibility_by_index_token(client, host, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    host(acc=_acc(name="NVIDIA GeForce GTX 1070"))
    d = client.get(URL).json()
    assert [(g["visible"], g["active"]) for g in d["devices"]] == [(False, False), (True, True)]


def test_cpu_host_reports_no_active_device(client, host, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cd, "_nvidia_rows", list)
    host(acc=_acc("cpu", "CPU"))
    d = client.get(URL).json()
    assert d["devices"] == [] and d["effective"]["is_gpu"] is False and d["effective"]["device"] == "cpu"


# ── PUT + persistence ────────────────────────────────────────────────────────
def test_put_gpu_persists_and_flags_restart(client, host) -> None:
    r = client.put(URL, json={"mode": "gpu", "device_id": ID_3090})
    assert r.status_code == 200
    d = r.json()
    assert d["saved"] == {"mode": "gpu", "device_id": ID_3090, "name": "NVIDIA GeForce RTX 3090", "available": True}
    # The process started on Auto, so a saved GPU is pending until the next start.
    assert d["restart_required"] is True
    assert saved_choice.load() == SavedChoice(mode="gpu", vendor="nvidia", uuid="GPU-297ed2f9-b4d6",
                                              name="NVIDIA GeForce RTX 3090", index=0)
    assert client.get(URL).json()["saved"] == d["saved"]  # round trip through the file


def test_put_goes_to_fts_root_only(client, host) -> None:
    client.put(URL, json={"mode": "cpu"})
    assert saved_choice.choice_path().is_file()
    assert saved_choice.choice_path().parent != Path.home() / ".finetune-studio"


@pytest.mark.parametrize("body", [
    {"mode": "gpu", "device_id": "nvidia:GPU-does-not-exist"},
    {"mode": "gpu"},
    {"mode": "gpu", "device_id": ""},
    {"mode": "auto", "device_id": ID_3090},
    {"mode": "cpu", "device_id": ID_3090},
])
def test_put_invalid_is_400_with_reason_and_saves_nothing(client, host, body: dict) -> None:
    r = client.put(URL, json=body)
    assert r.status_code == 400 and isinstance(r.json()["detail"], str) and r.json()["detail"]
    assert not saved_choice.choice_path().exists()


def test_put_unknown_device_names_the_valid_ids(client, host) -> None:
    detail = client.put(URL, json={"mode": "gpu", "device_id": "nvidia:nope"}).json()["detail"]
    assert "unknown device" in detail and ID_3090 in detail and ID_1070 in detail


def test_put_gpu_on_a_gpu_less_host_is_400(client, host, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cd, "_nvidia_rows", list)
    r = client.put(URL, json={"mode": "gpu", "device_id": ID_3090})
    assert r.status_code == 400 and "no selectable GPU" in r.json()["detail"]


@pytest.mark.parametrize("body", [{"mode": "turbo"}, {}, {"mode": 3}])
def test_put_schema_violation_is_422(client, host, body: dict) -> None:
    assert client.put(URL, json=body).status_code == 422


def test_put_unwritable_root_is_500_not_silent(client, host, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_k):
        raise PermissionError("read-only file system")
    monkeypatch.setattr(saved_choice, "save", boom)
    r = client.put(URL, json={"mode": "cpu"})
    assert r.status_code == 500 and "read-only" in r.json()["detail"]


# ── restart_required ─────────────────────────────────────────────────────────
def test_restart_cleared_once_the_process_started_with_the_saved_choice(client, host) -> None:
    gpu = SavedChoice(mode="gpu", vendor="nvidia", uuid="GPU-297ed2f9-b4d6", name="NVIDIA GeForce RTX 3090", index=0)
    saved_choice.save(gpu)
    host(env.AppliedPolicy("saved", gpu))
    d = client.get(URL).json()
    assert d["restart_required"] is False and d["effective"]["source"] == "saved"


def test_going_back_to_auto_needs_a_restart_too(client, host) -> None:
    gpu = SavedChoice(mode="gpu", vendor="nvidia", uuid="GPU-297ed2f9-b4d6", name="NVIDIA GeForce RTX 3090", index=0)
    host(env.AppliedPolicy("saved", gpu))
    saved_choice.save(gpu)
    assert client.put(URL, json={"mode": "auto"}).json()["restart_required"] is True


def test_resaving_the_same_choice_needs_no_restart(client, host) -> None:
    client.put(URL, json={"mode": "cpu"})
    host(env.AppliedPolicy("saved", SavedChoice(mode="cpu")), acc=_acc("cpu", "CPU"))
    assert client.put(URL, json={"mode": "cpu"}).json()["restart_required"] is False


# ── env override ─────────────────────────────────────────────────────────────
def test_env_override_wins_and_is_reported(client, host) -> None:
    host(env.AppliedPolicy("env", SavedChoice(), ("FTS_GPU_EXCLUDE",)))
    d = client.put(URL, json={"mode": "gpu", "device_id": ID_3090}).json()
    assert d["overridden_by_env"] is True and d["env_overrides"] == ["FTS_GPU_EXCLUDE"]
    assert d["restart_required"] is False  # a restart cannot lift an env var
    assert d["effective"]["source"] == "env"


def test_env_set_but_saved_auto_is_not_an_override(client, host) -> None:
    host(env.AppliedPolicy("env", SavedChoice(), ("FTS_GPU_EXCLUDE",)))
    d = client.get(URL).json()
    assert d["env_overrides"] == ["FTS_GPU_EXCLUDE"] and d["overridden_by_env"] is False


def test_saved_gpu_that_vanished_is_flagged(client, host) -> None:
    saved_choice.save(SavedChoice(mode="gpu", vendor="nvidia", uuid="GPU-gone", name="Old GPU", index=2))
    d = client.get(URL).json()
    assert d["saved"]["available"] is False and d["saved"]["name"] == "Old GPU"


def test_effective_note_surfaces_why_a_saved_gpu_was_not_applied(client, host) -> None:
    gpu = SavedChoice(mode="gpu", vendor="nvidia", uuid="GPU-gone", name="Old GPU", index=2)
    host(env.AppliedPolicy("saved", gpu, note="Saved GPU Old GPU is not present on this host; running with automatic selection."))
    assert "not present" in client.get(URL).json()["effective"]["note"]


# ── UI wiring ────────────────────────────────────────────────────────────────
def test_settings_page_has_the_card_and_script(client, host) -> None:
    html = client.get("/settings").text
    for needle in ('id="compute-card"', 'id="compute-select"', 'id="btn-compute-save"',
                   'data-testid="compute-restart-required"', "settings.js?v=10"):
        assert needle in html, needle
