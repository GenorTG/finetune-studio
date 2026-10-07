"""`fts dataset build-preference` — same pipeline as the API; the model is faked, nothing leaves tmp."""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from finetune_studio.cli import _registry
from finetune_studio.data.prep import preference as pref
from finetune_studio.db.datasets import list_datasets
from tests.preference_fakes import FakeModel, seed_project


def run_cli(monkeypatch, capsys, *argv: str) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "argv", ["fts", *argv])
    code = 0
    try:
        _registry.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    cap = capsys.readouterr()
    return code, cap.out, cap.err


@pytest.fixture
def fake_helper(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    model = FakeModel()

    @contextmanager
    def loaded():
        yield model

    monkeypatch.setattr(pref, "helper_loaded_for_cli", loaded)
    return model


def test_json_summary_by_project_name(fake_helper, monkeypatch, capsys) -> None:
    proj = seed_project()
    code, out, err = run_cli(monkeypatch, capsys, "dataset", "build-preference", "--project", "pref-docs",
                             "--max-pairs", "6", "--seed", "9", "--json")
    assert code == 0, err
    s = json.loads(out)
    assert s["pairs"] == 6 and s["by_kind"] == {"hallucination": 3, "abstain": 3}
    path = Path(s["dataset"]["path"])
    assert path.name == f"{proj['id']}-preference.jsonl" and len(path.read_text().splitlines()) == 6
    assert [d["data_path"] for d in list_datasets(proj["id"])] == [str(path)]
    assert err == "", "--json must keep stderr clean of progress"


def test_text_output_shows_counts_drops_and_length(fake_helper, monkeypatch, capsys) -> None:
    proj = seed_project()
    code, out, err = run_cli(monkeypatch, capsys, "dataset", "build-preference", "--project", proj["id"],
                             "--kinds", "hallucination", "--max-pairs", "4")
    assert code == 0, err
    assert "Pairs    : 4 (hallucination 4)" in out and "Length   :" in out
    assert "candidates tried" in err  # progress goes to stderr


def test_bad_kinds_and_unknown_project_exit_nonzero(fake_helper, monkeypatch, capsys) -> None:
    seed_project()
    code, _, err = run_cli(monkeypatch, capsys, "dataset", "build-preference", "--project", "pref-docs",
                           "--kinds", "style")
    assert code == 1 and "--kinds" in err and "style" in err
    code, _, err = run_cli(monkeypatch, capsys, "dataset", "build-preference", "--project", "ghost")
    assert code == 1 and "project not found" in err


def test_no_approved_pairs_exits_nonzero_with_the_reason(fake_helper, monkeypatch, capsys) -> None:
    seed_project(status="pending")
    code, _, err = run_cli(monkeypatch, capsys, "dataset", "build-preference", "--project", "pref-docs")
    assert code == 1 and "approved" in err.lower()


def test_unusable_output_is_reported_with_the_drop_reasons(monkeypatch, capsys) -> None:
    @contextmanager
    def loaded():
        yield FakeModel(hallucination="I don't know.")

    monkeypatch.setattr(pref, "helper_loaded_for_cli", loaded)
    proj = seed_project()
    code, _, err = run_cli(monkeypatch, capsys, "dataset", "build-preference", "--project", proj["id"],
                           "--kinds", "hallucination")
    assert code == 1 and "refusal_as_rejected" in err and list_datasets(proj["id"]) == []


# ── helper_loaded_for_cli: load when absent, unload what it loaded ─────

def test_helper_is_loaded_then_unloaded_when_none_is_resident(monkeypatch) -> None:
    events: list[str] = []
    state = {"resident": False}

    def fake_generate():
        return (lambda prompt: "x") if state["resident"] else None

    class Mgr:
        def load(self, pid, extra=None):
            events.append(f"load:{pid}")
            state["resident"] = True

    monkeypatch.setattr(pref, "helper_generate", fake_generate)
    monkeypatch.setattr("finetune_studio.models.manager.get_manager", lambda: Mgr())
    monkeypatch.setattr("finetune_studio.models.helper.missing_gguf_for_provider", lambda pid: "")
    monkeypatch.setattr("finetune_studio.models.llama_loader.unload_all_models", lambda: events.append("unload"))
    with pref.helper_loaded_for_cli() as generate:
        assert generate("q") == "x" and events == ["load:local-default"]
    assert events == ["load:local-default", "unload"]


def test_an_already_resident_helper_is_used_and_left_loaded(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(pref, "helper_generate", lambda: (lambda prompt: "y"))
    monkeypatch.setattr("finetune_studio.models.llama_loader.unload_all_models", lambda: events.append("unload"))
    with pref.helper_loaded_for_cli() as generate:
        assert generate("q") == "y"
    assert events == []


def test_missing_helper_gguf_is_an_honest_error(monkeypatch) -> None:
    monkeypatch.setattr(pref, "helper_generate", lambda: None)
    monkeypatch.setattr("finetune_studio.models.helper.missing_gguf_for_provider", lambda pid: "/nope/h.gguf")
    with pytest.raises(pref.NoGenerator, match="h.gguf"), pref.helper_loaded_for_cli():
        pass
