"""CLI error-path regressions found by the lane-C E2E sweep: no tracebacks, nonzero exit."""

from __future__ import annotations

import sys

import pytest

from finetune_studio.cli import _registry


def run_cli(monkeypatch, capsys, *argv: str) -> tuple[int, str]:
    monkeypatch.setattr(sys, "argv", ["fts", *argv])
    code = 0
    try:
        _registry.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    cap = capsys.readouterr()
    return code, cap.out + cap.err


@pytest.mark.parametrize("cmd", ["analyze", "optimize", "validate-hallucination"])
def test_missing_data_file_is_clean_error(monkeypatch, capsys, tmp_path, cmd):
    code, out = run_cli(monkeypatch, capsys, cmd, str(tmp_path / "nope.jsonl"))
    assert code == 1
    assert "Traceback" not in out
    assert "nope.jsonl" in out


def test_augment_missing_input_and_bad_output_dir(monkeypatch, capsys, tmp_path):
    code, out = run_cli(monkeypatch, capsys, "augment", str(tmp_path / "x.jsonl"),
                        "--output", str(tmp_path / "o.jsonl"))
    assert code == 1 and "Traceback" not in out
    src = tmp_path / "in.jsonl"
    src.write_text('{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}\n')
    code, out = run_cli(monkeypatch, capsys, "augment", str(src),
                        "--output", str(tmp_path / "nodir" / "o.jsonl"))
    assert code == 1 and "Traceback" not in out


def test_rag_ingest_missing_path_is_clean_error(monkeypatch, capsys, tmp_path):
    code, out = run_cli(monkeypatch, capsys, "rag", "ingest", str(tmp_path / "nope"),
                        "--store", str(tmp_path / "store"))
    assert code == 1 and "Traceback" not in out


def test_rag_remove_unknown_document_exits_nonzero(monkeypatch, capsys, tmp_path):
    code, out = run_cli(monkeypatch, capsys, "rag", "remove", "nonexist",
                        "--store", str(tmp_path / "store"))
    assert code == 1
    assert "not found" in out


def test_models_scans_same_dirs_as_webui(monkeypatch, capsys, tmp_path):
    from finetune_studio.config import settings

    seen: list = []
    import finetune_studio.models.registry as reg

    monkeypatch.setattr(reg, "scan_models", lambda dirs: seen.extend(dirs) or [])
    monkeypatch.setattr(settings, "model_dirs_extra", [str(tmp_path / "extra")])
    run_cli(monkeypatch, capsys, "models")
    assert str(tmp_path / "extra") in seen


# ── suite / compare / benchmark ──────────────────────────────────────────────


class _BoomEngine:
    def __init__(self, *a, **k):
        raise AssertionError("model must not load before the suite/args are validated")


@pytest.mark.parametrize("body", ["not json", '{"cases": "x"}', '{"cases": [{"name": "c", "prompt": "p"}]}', "[]"])
def test_suite_bad_file_fails_before_model_load(monkeypatch, capsys, tmp_path, body):
    import finetune_studio.testing.inference as inf

    monkeypatch.setattr(inf, "InferenceEngine", _BoomEngine)
    model = tmp_path / "m"
    model.mkdir()
    suite = tmp_path / "s.json"
    suite.write_text(body)
    code, out = run_cli(monkeypatch, capsys, "suite", str(model), str(suite))
    assert code == 1 and "Traceback" not in out and "Error" in out


def test_benchmark_unknown_suite_fails_before_model_load(monkeypatch, capsys, tmp_path):
    import finetune_studio.testing.inference as inf

    monkeypatch.setattr(inf, "InferenceEngine", _BoomEngine)
    model = tmp_path / "m"
    model.mkdir()
    code, out = run_cli(monkeypatch, capsys, "benchmark", str(model), "--suite", "bogus")
    assert code == 2 and "unknown benchmark" in out


def test_install_diagnose_driver_mapping_mirrors_install_sh() -> None:
    """The driver->CUDA mapping lives only in scripts/accel_plan.py; both installers delegate."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sh = (root / "install.sh").read_text()
    py = (root / "scripts" / "install_diagnose.py").read_text()
    assert "accel_plan" in sh and "accel_plan" in py
    assert not re.search(r'CUDA_VER="cu\d+"', sh)
    assert not re.search(r'cuda_ver = "cu\d+"', py)


def test_update_sh_check_mode_does_not_pull_or_migrate() -> None:
    """`update.sh --check` is documented as a dry-run: pull and init_db must be gated."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "update.sh").read_text()
    pull = src.index("git pull --ff-only")
    assert 'CHECK_MODE" = "1"' in src[:pull]
    mig = src.index("init_db()")
    assert 'CHECK_MODE" = "1"' in src[mig - 400 : mig]
