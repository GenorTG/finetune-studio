"""Command-level regression tests for CLI truthfulness.

Audit finding (docs/audit/APP-AUDIT-2026-10-02.md, "CLI truthfulness"):
options accepted but ignored, ``fts suite`` scoring an unjudged run, and
``fts validate`` printing invalid while exiting zero. Every test drives the
real entry point (``cli.main`` with a patched ``sys.argv``) so the parser,
registry and handler are exercised together.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import sys
from pathlib import Path
from typing import ClassVar

import pytest

from finetune_studio.cli import _registry
from finetune_studio.cli._parser import build_parser


def run_cli(monkeypatch, capsys, *argv: str) -> tuple[int, str]:
    """Run ``fts <argv>`` through ``main()``; return (exit code, stdout+stderr)."""
    monkeypatch.setattr(sys, "argv", ["fts", *argv])
    code = 0
    try:
        _registry.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    captured = capsys.readouterr()
    return code, captured.out + captured.err


# ── fts validate: exit status must match the verdict ─────────────────────────


def test_validate_invalid_file_exits_nonzero(monkeypatch, capsys, tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    code, out = run_cli(monkeypatch, capsys, "validate", str(bad))
    assert "❌" in out
    assert code != 0


def test_validate_missing_file_exits_nonzero(monkeypatch, capsys, tmp_path: Path) -> None:
    code, out = run_cli(monkeypatch, capsys, "validate", str(tmp_path / "nope.jsonl"))
    assert "File not found" in out
    assert code != 0


def test_validate_valid_file_exits_zero(monkeypatch, capsys, tmp_path: Path) -> None:
    good = tmp_path / "good.jsonl"
    good.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"},
                                 {"role": "assistant", "content": "hello"}]}) + "\n",
        encoding="utf-8",
    )
    code, out = run_cli(monkeypatch, capsys, "validate", str(good))
    assert "✅" in out
    assert code == 0


def test_validate_reports_every_file_then_fails(monkeypatch, capsys, tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    good = tmp_path / "good.jsonl"
    good.write_text(
        json.dumps({"messages": [{"role": "user", "content": "q"},
                                 {"role": "assistant", "content": "a"}]}) + "\n",
        encoding="utf-8",
    )
    code, out = run_cli(monkeypatch, capsys, "validate", str(bad), str(good))
    assert "bad.jsonl" in out and "good.jsonl" in out  # no early abort
    assert code != 0


# ── fts suite: never score an unjudged collection ────────────────────────────


class _FakeEngine:
    """Stand-in for InferenceEngine: answers per-question from a class table."""

    answers: ClassVar[dict[str, str]] = {}
    fail: ClassVar[bool] = False

    def load(self, _path: str) -> None:
        pass

    def unload(self) -> None:
        pass

    def generate(self, messages, **_kw) -> str:
        if self.fail:
            raise RuntimeError("engine crashed")
        return self.answers.get(messages[-1]["content"], "")


@pytest.fixture
def suite_env(monkeypatch, tmp_path: Path):
    from finetune_studio.testing import inference

    _FakeEngine.answers = {}
    _FakeEngine.fail = False
    monkeypatch.setattr(inference, "InferenceEngine", _FakeEngine)
    model = tmp_path / "model.gguf"
    model.write_text("x", encoding="utf-8")
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps([
        {"name": "capital", "question": "Capital of France?",
         "correct_answer": "Paris", "keywords": ["Paris"]},
        {"name": "color", "question": "Color of grass?",
         "correct_answer": "green", "keywords": ["green"]},
    ]), encoding="utf-8")
    return model, suite


def test_suite_judges_results_before_scoring(monkeypatch, capsys, suite_env) -> None:
    model, suite = suite_env
    _FakeEngine.answers = {"Capital of France?": "It is Paris.",
                           "Color of grass?": "blue"}
    code, out = run_cli(monkeypatch, capsys, "suite", str(model), str(suite))
    assert code == 0, out
    assert "Pass rate: 50.0% (1/2)" in out


def test_suite_json_includes_verdicts(monkeypatch, capsys, suite_env) -> None:
    model, suite = suite_env
    _FakeEngine.answers = {"Capital of France?": "Paris", "Color of grass?": "green"}
    code, out = run_cli(monkeypatch, capsys, "suite", str(model), str(suite), "--json")
    assert code == 0, out
    payload = json.loads(out[out.index("{"):])
    assert [r["verdict"] for r in payload["results"]] == ["pass", "pass"]
    assert payload["scores"]["judged"] == 2
    assert payload["scores"]["pass_rate"] == 100.0


def test_suite_with_no_judged_cases_fails_without_score(monkeypatch, capsys, suite_env) -> None:
    model, suite = suite_env
    _FakeEngine.fail = True  # every case errors -> no verdicts at all
    code, out = run_cli(monkeypatch, capsys, "suite", str(model), str(suite))
    assert code != 0
    assert "Pass rate" not in out
    assert "unjudged" in out.lower()


def test_suite_json_with_no_judged_cases_reports_no_score(monkeypatch, capsys, suite_env) -> None:
    model, suite = suite_env
    _FakeEngine.fail = True
    code, out = run_cli(monkeypatch, capsys, "suite", str(model), str(suite), "--json")
    assert code != 0
    payload = json.JSONDecoder().raw_decode(out[out.index("{"):])[0]
    assert payload["scores"] is None
    assert payload["scores_unavailable"]


# ── accepted-but-ignored flags are removed from the parser ───────────────────


@pytest.mark.parametrize(
    "argv",
    [
        ("analyze", "data.jsonl", "--fix"),
        ("analyze", "data.jsonl", "--output", "out.jsonl"),
        ("augment", "data.jsonl", "--output", "o.jsonl", "--count", "10"),
        ("augment", "data.jsonl", "--output", "o.jsonl", "--ratio", "0.5"),
        ("benchmark", "model.gguf", "--max-tokens", "8"),
        ("benchmark", "model.gguf", "--temperature", "0.5"),
        ("benchmark", "model.gguf", "--real"),
        ("compare", "suite.json", "--models", "a=b", "--real"),
    ],
)
def test_no_op_flags_are_rejected(monkeypatch, capsys, argv: tuple[str, ...]) -> None:
    code, out = run_cli(monkeypatch, capsys, *argv)
    assert code == 2
    assert "unrecognized arguments" in out


def _handler_module_source(command: str) -> str:
    handler = _registry.COMMANDS[command]
    return inspect.getsource(importlib.import_module(handler.__module__))


def _subparser_flags(parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    """(command-path, dest) for every real option under every subcommand."""
    found: list[tuple[str, str]] = []

    def walk(p: argparse.ArgumentParser, path: str) -> None:
        for action in p._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    walk(sub, f"{path} {name}".strip())
            elif action.dest not in ("help", argparse.SUPPRESS) and path:
                found.append((path, action.dest))

    walk(parser, "")
    return found


def test_every_parser_option_is_read_by_its_handler() -> None:
    """A flag that no handler line reads is a silent no-op (the audit finding)."""
    unread = []
    for path, dest in _subparser_flags(build_parser()):
        command = path.split()[0]
        source = _handler_module_source(command)
        pattern = rf"\bargs\.{re.escape(dest)}\b|[\"']{re.escape(dest)}[\"']"
        if not re.search(pattern, source):
            unread.append(f"{path}: {dest}")
    assert not unread, f"parser options never read by their handler: {unread}"


# ── fts augment: --type values must reach the augmenter ──────────────────────


def _write_jsonl(path: Path) -> Path:
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "q"},
                                 {"role": "assistant", "content": "a"}]}) + "\n",
        encoding="utf-8",
    )
    return path


def test_augment_type_aliases_reach_generators(monkeypatch, capsys, tmp_path: Path) -> None:
    from finetune_studio.training import data_augmentation

    seen: dict[str, list[str]] = {}

    def fake_augment(self, data, weaknesses=None):
        seen["weaknesses"] = list(weaknesses or [])
        return list(data)

    monkeypatch.setattr(data_augmentation.DataAugmenter, "augment_dataset", fake_augment)
    src = _write_jsonl(tmp_path / "in.jsonl")
    out_path = tmp_path / "out.jsonl"
    code, out = run_cli(monkeypatch, capsys, "augment", str(src), "--output", str(out_path),
                        "--type", "language,persona")
    assert code == 0, out
    assert {"language_balance", "persona_preservation"} <= set(seen["weaknesses"])
    assert "language" not in seen["weaknesses"]  # alias must not leak through


def test_augment_unknown_type_fails_instead_of_being_ignored(monkeypatch, capsys, tmp_path: Path) -> None:
    src = _write_jsonl(tmp_path / "in.jsonl")
    out_path = tmp_path / "out.jsonl"
    code, out = run_cli(monkeypatch, capsys, "augment", str(src), "--output", str(out_path),
                        "--type", "bogus")
    assert code != 0
    assert "bogus" in out
    assert not out_path.exists()


# ── fts benchmark: output must not claim settings it does not apply ──────────


def test_benchmark_output_does_not_claim_temperature(monkeypatch, capsys, tmp_path: Path) -> None:
    import finetune_studio.benchmarks.real_benchmarks as rb
    from finetune_studio.testing import inference

    class _Engine:
        def load(self, _p: str) -> None:
            pass

        def unload(self) -> None:
            pass

    def fake_run_all(self, engine, **_kw):
        return {"benchmarks": {}, "summary": {"total_correct": 0, "total_questions": 0,
                                              "overall_accuracy": 0.0}}

    monkeypatch.setattr(inference, "InferenceEngine", _Engine)
    monkeypatch.setattr(rb.RealBenchmarkSuite, "run_all", fake_run_all)
    model = tmp_path / "m.gguf"
    model.write_text("x", encoding="utf-8")
    code, out = run_cli(monkeypatch, capsys, "benchmark", str(model))
    assert code == 0, out
    assert "Temperature:" not in out
