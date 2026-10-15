"""Regression: test-judging settings, HF_HOME honoring, tokenizer/cache warnings."""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from finetune_studio import hf_env


@pytest.fixture()
def settings_file(tmp_path: Path, monkeypatch):
    from finetune_studio.webui.routes import settings as settings_mod

    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", tmp_path / "settings.json")
    return settings_mod


# ---------------------------------------------------------------- test-judging settings
def test_auto_judge_is_off_and_the_judge_is_the_helper_seat_by_default(client, settings_file) -> None:
    assert settings_file.get_test_settings() == {"auto_judge": False, "judge_provider_id": ""}
    body = client.get("/api/settings/testing").json()
    assert body["auto_judge"] is False and body["judge_provider_id"] == ""
    assert body["effective_judge_provider_id"]  # '' resolves to the helper seat
    seat = [p for p in body["providers"] if p["is_helper_seat"]]
    assert len(seat) == 1 and body["providers"][0]["id"] == seat[0]["id"]  # seat listed first
    assert client.get("/api/settings").json()["testing"] == {"auto_judge": False, "judge_provider_id": ""}


def test_testing_settings_roundtrip_and_partial_updates(client, settings_file) -> None:
    provider = client.get("/api/settings/testing").json()["providers"][0]["id"]
    r = client.put("/api/settings/testing", json={"auto_judge": True, "judge_provider_id": provider})
    assert r.status_code == 200, r.text
    assert r.json()["auto_judge"] is True and r.json()["effective_judge_provider_id"] == provider
    assert settings_file.get_test_settings() == {"auto_judge": True, "judge_provider_id": provider}
    # a partial update leaves the other field alone; '' clears the judge back to the helper seat
    client.put("/api/settings/testing", json={"auto_judge": False})
    assert settings_file.get_test_settings() == {"auto_judge": False, "judge_provider_id": provider}
    client.put("/api/settings/testing", json={"judge_provider_id": ""})
    assert settings_file.get_test_settings()["judge_provider_id"] == ""


@pytest.mark.parametrize("body,fragment", [
    ({"judge_provider_id": "no-such-provider"}, "unknown provider"),
    ({"auto_judge": "yes"}, "auto_judge must be true or false"),
    ({"auto_judge": 1}, "auto_judge must be true or false"),
    ({"auto_judge": None}, "auto_judge must be true or false"),
])
def test_testing_settings_reject_bad_values_and_save_nothing(client, settings_file, body, fragment) -> None:
    r = client.put("/api/settings/testing", json=body)
    assert r.status_code == 400 and fragment in r.json()["detail"]
    assert settings_file.get_test_settings() == {"auto_judge": False, "judge_provider_id": ""}


@pytest.mark.parametrize("raw", ["[1]", "not json"])
def test_testing_settings_reject_a_non_object_body(client, settings_file, raw) -> None:
    r = client.put("/api/settings/testing", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_the_scripted_judge_api_key_settings_are_gone(settings_file) -> None:
    for name in ("get_judge_config", "JUDGE_MODES", "judge_config_public"):
        assert not hasattr(settings_file, name)
    from finetune_studio.webui.routes import benchmarks

    assert not hasattr(benchmarks, "_apply_configured_judge")
    src_root = Path(__file__).resolve().parents[1] / "src" / "finetune_studio"
    offenders = [str(p.relative_to(src_root)) for p in src_root.rglob("*.py") if "FTS_JUDGE_" in p.read_text()]
    assert not offenders, offenders


def test_a_legacy_judge_api_key_in_settings_json_is_never_echoed(client, settings_file) -> None:
    settings_file._save({"judge_api_key": "sk-legacy", "judge_mode": "ai"})
    assert "sk-legacy" not in client.get("/api/settings").text
    assert "sk-legacy" not in client.patch("/api/settings", json={"port": 7861}).text


# ------------------------------------------------------------------- HF paths
def test_hf_paths_honor_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert hf_env.hf_home() == tmp_path / "hf"
    assert hf_env.hf_hub_cache() == tmp_path / "hf" / "hub"
    assert hf_env.hf_cache_source() == "HF_HOME"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub2"))
    assert hf_env.hf_hub_cache() == tmp_path / "hub2"
    assert hf_env.is_in_hub_cache(str(tmp_path / "hub2" / "models--a--b"))
    assert not hf_env.is_in_hub_cache(str(tmp_path / "elsewhere"))


def test_hf_cache_dir_honors_hf_home_and_sets_no_transformers_cache(
    monkeypatch, tmp_path: Path
) -> None:
    from finetune_studio.data import shared_models

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    assert shared_models.hf_cache_dir() == tmp_path / "hf"
    assert "TRANSFORMERS_CACHE" not in os.environ


def test_debug_info_reports_effective_cache(client, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    paths = client.get("/api/debug/info").json()["paths"]
    assert paths["hf_cache"] == str(tmp_path / "hf")
    assert paths["hf_hub_cache"] == str(tmp_path / "hf" / "hub")
    assert paths["hf_cache_source"] == "HF_HOME"


def test_no_hardcoded_hf_cache_paths_left() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "finetune_studio"
    offenders = []
    for p in root.rglob("*.py"):
        if p.name == "hf_env.py":
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if '".cache" / "huggingface"' in line or (
                "TRANSFORMERS_CACHE" in line and not line.lstrip().startswith("#")
            ):
                offenders.append(f"{p.relative_to(root)}:{i}")
    assert not offenders, offenders


# ------------------------------------------------------------------ tokenizer
def test_load_tokenizer_passes_fix_mistral_regex(monkeypatch) -> None:
    calls: list[dict] = []

    class Tok:
        @staticmethod
        def from_pretrained(path, **kw):
            calls.append(kw)
            return object()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=Tok))
    hf_env.load_tokenizer("m")
    assert calls[0]["fix_mistral_regex"] is True and calls[0]["trust_remote_code"] is True


def test_load_tokenizer_falls_back_when_kwarg_rejected(monkeypatch) -> None:
    calls: list[dict] = []

    class Tok:
        @staticmethod
        def from_pretrained(path, **kw):
            calls.append(kw)
            if "fix_mistral_regex" in kw:
                raise TypeError("unexpected kwarg")
            return "tok"

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=Tok))
    assert hf_env.load_tokenizer("m") == "tok"
    assert len(calls) == 2


def test_tokenizer_call_sites_use_helper() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "finetune_studio"
    bad = [str(p.relative_to(root)) for p in root.rglob("*.py")
           if p.name != "hf_env.py" and "AutoTokenizer.from_pretrained" in p.read_text()]
    assert not bad, bad
