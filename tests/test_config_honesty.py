"""Regression: UI-driven judge config, HF_HOME honoring, tokenizer/cache warnings."""
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
    for k in ("FTS_JUDGE_MODEL", "FTS_JUDGE_API", "FTS_JUDGE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return settings_mod


# ---------------------------------------------------------------- judge config
def test_judge_default_mode_follows_env_key(settings_file, monkeypatch) -> None:
    assert settings_file.get_judge_config()["mode"] == "heuristic"
    monkeypatch.setenv("FTS_JUDGE_API_KEY", "sk-env")
    cfg = settings_file.get_judge_config()
    assert cfg["mode"] == "ai" and cfg["api_key"] == "sk-env"


def test_judge_settings_roundtrip_never_echoes_key(client, settings_file) -> None:
    r = client.patch("/api/settings", json={
        "judge_mode": "ai", "judge_model": "my-judge",
        "judge_api_url": "http://localhost:1234/v1/", "judge_api_key": "sk-secret",
    })
    assert r.status_code == 200
    assert "sk-secret" not in r.text
    assert r.json()["judge"] == {
        "mode": "ai", "model": "my-judge",
        "api_url": "http://localhost:1234/v1", "api_key_set": True,
    }
    for url in ("/api/settings", "/api/settings/judge"):
        assert "sk-secret" not in client.get(url).text
    assert settings_file.get_judge_config()["api_key"] == "sk-secret"
    # blank key keeps the stored one; explicit clear removes it
    client.patch("/api/settings", json={"judge_api_key": ""})
    assert settings_file.get_judge_config()["api_key"] == "sk-secret"
    r = client.patch("/api/settings", json={"judge_api_key_clear": True})
    assert r.json()["judge"]["api_key_set"] is False


def test_judge_mode_validated(client, settings_file) -> None:
    assert client.patch("/api/settings", json={"judge_mode": "bogus"}).status_code == 400


def test_run_applies_configured_ai_judge(settings_file, monkeypatch) -> None:
    from finetune_studio.testing import judge as judge_mod
    from finetune_studio.testing.suite import CaseResult
    from finetune_studio.webui.routes import benchmarks

    settings_file._save({
        "judge_mode": "ai", "judge_model": "m1",
        "judge_api_url": "http://x/v1", "judge_api_key": "k",
    })
    seen: dict = {}

    def fake_ai(q, c, a, model, api_url, api_key):
        seen.update(model=model, api_url=api_url, api_key=api_key)
        return "pass", "ok", 0.9

    monkeypatch.setattr(judge_mod, "judge_case_ai", fake_ai)
    r = CaseResult(case_name="c", category="x", question="q", correct_answer="Paris",
                   model_answer="Paris", time_ms=1)
    benchmarks._apply_configured_judge([r], "ai")
    assert (r.verdict, r.judge, r.judge_model) == ("pass", "ai", "m1")
    assert seen == {"model": "m1", "api_url": "http://x/v1", "api_key": "k"}


def test_ai_judge_without_key_leaves_cases_for_heuristic(settings_file) -> None:
    from finetune_studio.testing.suite import CaseResult
    from finetune_studio.webui.routes import benchmarks

    r = CaseResult(case_name="c", category="x", question="q", correct_answer="a",
                   model_answer="a", time_ms=1)
    benchmarks._apply_configured_judge([r], "ai")
    assert r.verdict == ""


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
