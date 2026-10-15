"""Persisted compute-device choice: file round trip and how ``accel.env`` applies it at start-up.

Hermetic: injected environments, fake nvidia-smi/rocm-smi runners, a tmp file — no real GPU, no
real ``$FTS_ROOT``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from finetune_studio.accel import env, saved_choice
from finetune_studio.accel.saved_choice import SavedChoice

SMI = ("0, GPU-297ed2f9-b4d6, NVIDIA GeForce RTX 3090\n"
       "1, GPU-b6f0e3aa-5f4b, NVIDIA GeForce GTX 1070\n")
ROCM = json.dumps({"card0": {"Card Series": "Radeon RX 7900 XTX"}, "card1": {"Card Series": "Radeon RX 6600"}})
GPU_3090 = SavedChoice(mode="gpu", vendor="nvidia", uuid="GPU-297ed2f9-b4d6", name="NVIDIA GeForce RTX 3090", index=0)


@pytest.fixture(autouse=True)
def _restore_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "_applied", env.AppliedPolicy("default", SavedChoice()))


def _apply(environ: dict[str, str], path: Path, smi: str = SMI, rocm: str = "") -> dict[str, str]:
    return env.apply_device_policy(environ, nvidia=lambda _c: smi, amd=lambda _c: rocm, saved_path=path)


# ── persistence ──────────────────────────────────────────────────────────────
def test_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "compute_device.json"
    saved_choice.save(GPU_3090, p)
    assert saved_choice.load(p) == GPU_3090
    assert not list(p.parent.glob("*.tmp"))  # atomic write leaves no temp file
    saved_choice.save(SavedChoice(mode="cpu"), p)
    assert saved_choice.load(p) == SavedChoice(mode="cpu")


def test_missing_file_is_auto(tmp_path: Path) -> None:
    assert saved_choice.load(tmp_path / "nope.json") == SavedChoice()


@pytest.mark.parametrize("content", [
    "not json", "[]", '{"mode": "turbo"}', '{"mode": "gpu"}',
    '{"mode": "gpu", "vendor": "intel", "uuid": "x"}', '{"mode": "gpu", "vendor": "nvidia"}',
])
def test_corrupt_or_incomplete_file_is_auto(tmp_path: Path, content: str) -> None:
    p = tmp_path / "c.json"
    p.write_text(content)
    assert saved_choice.load(p) == SavedChoice()  # a bad file must never block start-up


def test_path_follows_fts_root(tmp_path: Path) -> None:
    assert saved_choice.choice_path({"FTS_ROOT": str(tmp_path)}) == tmp_path / "compute_device.json"
    assert saved_choice.choice_path({}).name == "compute_device.json"


def test_device_id_only_for_gpu_mode() -> None:
    assert GPU_3090.device_id == "nvidia:GPU-297ed2f9-b4d6"
    assert SavedChoice(mode="cpu").device_id == "" and SavedChoice().device_id == ""


# ── env overrides ────────────────────────────────────────────────────────────
def test_env_overrides_lists_explicit_policy_vars() -> None:
    assert saved_choice.env_overrides({}) == ()
    assert saved_choice.env_overrides({"FTS_DEVICE": "auto"}) == ()
    assert saved_choice.env_overrides({"FTS_DEVICE": ""}) == ()
    assert saved_choice.env_overrides({"FTS_DEVICE": "cuda:1", "FTS_GPU_EXCLUDE": "GTX 1070",
                                       "CUDA_VISIBLE_DEVICES": ""}) == (
        "FTS_GPU_EXCLUDE", "FTS_DEVICE", "CUDA_VISIBLE_DEVICES")


# ── applying at start-up ─────────────────────────────────────────────────────
def test_saved_gpu_masks_the_other_card(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    saved_choice.save(GPU_3090, p)
    e: dict[str, str] = {}
    assert _apply(e, p) == {"CUDA_VISIBLE_DEVICES": "GPU-297ed2f9-b4d6"}
    assert e["CUDA_VISIBLE_DEVICES"] == "GPU-297ed2f9-b4d6"
    applied = env.get_applied()
    assert applied.source == "saved" and applied.saved == GPU_3090 and applied.note == ""


def test_saved_gpu_survives_index_reordering(tmp_path: Path) -> None:
    """The UUID is the identity; the saved index is only a hint."""
    p = tmp_path / "c.json"
    saved_choice.save(GPU_3090, p)
    swapped = ("0, GPU-b6f0e3aa-5f4b, NVIDIA GeForce GTX 1070\n"
               "1, GPU-297ed2f9-b4d6, NVIDIA GeForce RTX 3090\n")
    assert _apply({}, p, smi=swapped) == {"CUDA_VISIBLE_DEVICES": "GPU-297ed2f9-b4d6"}


def test_saved_gpu_falls_back_to_unique_name_when_uuid_changed(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    saved_choice.save(GPU_3090, p)
    replaced = ("0, GPU-new-uuid-0001, NVIDIA GeForce RTX 3090\n"
                "1, GPU-b6f0e3aa-5f4b, NVIDIA GeForce GTX 1070\n")
    assert _apply({}, p, smi=replaced) == {"CUDA_VISIBLE_DEVICES": "GPU-new-uuid-0001"}


def test_saved_amd_gpu_sets_hip_visible_devices(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    saved_choice.save(SavedChoice(mode="gpu", vendor="amd", uuid="1", name="Radeon RX 6600", index=1), p)
    assert _apply({}, p, smi="", rocm=ROCM) == {"HIP_VISIBLE_DEVICES": "1"}


def test_saved_gpu_missing_runs_auto_and_says_why(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    saved_choice.save(GPU_3090, p)
    e: dict[str, str] = {}
    only_1070 = "0, GPU-b6f0e3aa-5f4b, NVIDIA GeForce GTX 1070\n"
    assert _apply(e, p, smi=only_1070) == {}
    assert "CUDA_VISIBLE_DEVICES" not in e  # never hides a card it cannot account for
    assert "RTX 3090" in env.get_applied().note and "not present" in env.get_applied().note


def test_saved_only_gpu_of_its_vendor_masks_nothing(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    saved_choice.save(GPU_3090, p)
    e: dict[str, str] = {}
    assert _apply(e, p, smi="0, GPU-297ed2f9-b4d6, NVIDIA GeForce RTX 3090\n") == {}
    assert e == {} and env.get_applied().source == "saved"


def test_saved_cpu_hides_every_gpu_and_forces_cpu(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    saved_choice.save(SavedChoice(mode="cpu"), p)
    e: dict[str, str] = {}
    _apply(e, p)
    # Masking is what keeps llama.cpp (n_gpu_layers=-1) off the cards, FTS_DEVICE steers torch.
    assert e == {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "FTS_DEVICE": "cpu"}


def test_saved_auto_changes_nothing(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    saved_choice.save(SavedChoice(), p)
    e: dict[str, str] = {}
    assert _apply(e, p) == {} and e == {}
    assert env.get_applied().source == "default"


@pytest.mark.parametrize("var,value", [
    ("FTS_GPU_EXCLUDE", "GTX 1070"), ("FTS_GPU_DEVICES", "1"), ("FTS_DEVICE", "cuda:1"),
    ("CUDA_VISIBLE_DEVICES", "1"), ("HIP_VISIBLE_DEVICES", "0"), ("ROCR_VISIBLE_DEVICES", "0"),
])
def test_explicit_env_beats_saved_choice(tmp_path: Path, var: str, value: str) -> None:
    p = tmp_path / "c.json"
    saved_choice.save(SavedChoice(mode="cpu"), p)
    e = {var: value}
    _apply(e, p)
    assert e.get("FTS_DEVICE") == ("cuda:1" if var == "FTS_DEVICE" else None)  # saved cpu not applied
    assert e.get("HIP_VISIBLE_DEVICES") != ""
    applied = env.get_applied()
    assert applied.source == "env" and applied.env_overrides == (var,) and applied.saved == SavedChoice(mode="cpu")


def test_env_exclude_still_applies_next_to_a_saved_choice(tmp_path: Path) -> None:
    """The live host's FTS_GPU_EXCLUDE drop-in keeps masking the GTX 1070 whatever was saved."""
    p = tmp_path / "c.json"
    saved_choice.save(SavedChoice(mode="gpu", vendor="nvidia", uuid="GPU-b6f0e3aa-5f4b",
                                  name="NVIDIA GeForce GTX 1070", index=1), p)
    e = {"FTS_GPU_EXCLUDE": "GTX 1070"}
    assert _apply(e, p) == {"CUDA_VISIBLE_DEVICES": "GPU-297ed2f9-b4d6"}


def test_injected_environment_does_not_read_a_file_unless_given_its_path(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FTS_ROOT", str(tmp_path))
    saved_choice.save(SavedChoice(mode="cpu"), tmp_path / "compute_device.json")
    e: dict[str, str] = {}
    env.apply_device_policy(e, nvidia=lambda _c: SMI, amd=lambda _c: "")
    assert e == {}  # hermetic: tests that inject an environment never see a developer's saved file


def test_process_environment_reads_the_saved_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``environ=None`` is the import-time path: it must find ``$FTS_ROOT/compute_device.json``."""
    for var in (*saved_choice.ENV_POLICY_VARS, "FTS_ROOT"):
        monkeypatch.setenv(var, "x")   # setenv-then-delenv registers the undo that restores "absent"
        monkeypatch.delenv(var)
    monkeypatch.setenv("FTS_ROOT", str(tmp_path))
    saved_choice.save(GPU_3090, tmp_path / "compute_device.json")
    seen = env.apply_device_policy(nvidia=lambda _c: SMI, amd=lambda _c: "")
    assert seen == {"CUDA_VISIBLE_DEVICES": "GPU-297ed2f9-b4d6"}
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-297ed2f9-b4d6"
