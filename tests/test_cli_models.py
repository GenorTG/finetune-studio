"""CLI parity for Hugging Face search, persistent downloads and local deletion."""
from __future__ import annotations

import json
import sys

import pytest

from finetune_studio.cli import _registry


def run_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "argv", ["fts", *argv])
    try:
        _registry.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    else:
        code = 0
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_search_uses_same_hub_results_and_json_shape(monkeypatch, capsys) -> None:
    from finetune_studio.webui.routes import hf_models

    monkeypatch.setattr(hf_models, "_search_hf", lambda req: [{
        "repo_id": "Org/Tiny", "downloads": 12, "likes": 3, "tags": ["transformers"],
    }])
    code, stdout, stderr = run_cli(monkeypatch, capsys, "models", "search", "Tiny", "--json")
    assert code == 0 and not stderr
    data = json.loads(stdout)
    assert data["query"] == "Tiny" and data["results"][0]["repo_id"] == "Org/Tiny"


def test_search_reports_hub_unavailable(monkeypatch, capsys) -> None:
    from finetune_studio.webui.routes import hf_models

    def unavailable(_req):
        raise hf_models.HFUnavailableError("offline")

    monkeypatch.setattr(hf_models, "_search_hf", unavailable)
    code, stdout, stderr = run_cli(monkeypatch, capsys, "models", "search", "Tiny")
    assert code == 1 and not stdout and "Hugging Face Hub is unreachable" in stderr


def test_download_queues_shared_persistent_job(temp_db: None, monkeypatch, capsys) -> None:
    from finetune_studio.webui import download_jobs

    seen = {}

    def start(**kwargs):
        seen.update(kwargs)
        return {"id": "job-1", "started_at": "now"}

    monkeypatch.setattr(download_jobs, "start", start)
    code, stdout, stderr = run_cli(monkeypatch, capsys, "models", "download", "Org/Tiny", "--file", "m.safetensors", "--json")
    assert code == 0 and not stderr
    assert seen["repo_id"] == "Org/Tiny" and seen["filename"] == "m.safetensors"
    assert seen["revision"] == "main" and seen["dest"].endswith("Org__Tiny")
    assert json.loads(stdout)["job_id"] == "job-1"


def test_download_status_and_cancel_report_unknown_ids(temp_db: None, monkeypatch, capsys) -> None:
    from finetune_studio.webui import download_jobs

    monkeypatch.setattr(download_jobs, "progress", lambda _job_id: {"status": "downloading", "repo_id": "Org/Tiny",
                                                                       "bytes_done": 5, "bytes_total": 10})
    code, stdout, stderr = run_cli(monkeypatch, capsys, "models", "download-status", "job-1", "--json")
    assert code == 0 and not stderr and json.loads(stdout)["bytes_done"] == 5
    monkeypatch.setattr(download_jobs, "progress", lambda _job_id: None)
    code, stdout, stderr = run_cli(monkeypatch, capsys, "models", "download-status", "missing")
    assert code == 1 and not stdout and "job not found" in stderr

    monkeypatch.setattr(download_jobs, "cancel", lambda _job_id: "cancelled")
    code, stdout, stderr = run_cli(monkeypatch, capsys, "models", "cancel-download", "job-1", "--json")
    assert code == 0 and not stderr and json.loads(stdout)["status"] == "cancelled"


def test_delete_requires_yes_in_noninteractive_use(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    code, stdout, stderr = run_cli(monkeypatch, capsys, "models", "delete", "Org/Tiny")
    assert code == 1 and not stdout and "requires --yes" in stderr


def test_delete_removes_only_downloaded_cache_and_keeps_legacy_listing(temp_db: None, monkeypatch, capsys, tmp_path) -> None:
    from finetune_studio.inference.remote import RemoteEngine
    from finetune_studio.models import hf_cache
    from finetune_studio.webui.routes import hf_models

    monkeypatch.setattr(hf_models, "_LOCAL", tmp_path)
    monkeypatch.setattr(hf_cache, "_LOCAL", tmp_path)
    model = tmp_path / "Org__Tiny"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(RemoteEngine, "_status", lambda _self, fresh=False: {"loaded": False})

    code, stdout, stderr = run_cli(monkeypatch, capsys, "models", "delete", "Org/Tiny", "--yes", "--json")
    assert code == 0 and not stderr and json.loads(stdout)["deleted"] == "Org/Tiny"
    assert not model.exists()


def test_delete_refuses_a_loaded_model(temp_db: None, monkeypatch, capsys, tmp_path) -> None:
    from finetune_studio.inference.remote import RemoteEngine
    from finetune_studio.models import hf_cache
    from finetune_studio.webui.routes import hf_models

    monkeypatch.setattr(hf_models, "_LOCAL", tmp_path)
    monkeypatch.setattr(hf_cache, "_LOCAL", tmp_path)
    model = tmp_path / "Org__Tiny"
    model.mkdir()
    loaded = model / "model.safetensors"
    monkeypatch.setattr(RemoteEngine, "_status", lambda _self, fresh=False: {
        "loaded": True, "model_path": str(loaded),
    })
    code, stdout, stderr = run_cli(monkeypatch, capsys, "models", "delete", "Org/Tiny", "--yes")
    assert code == 1 and not stdout and "currently loaded" in stderr
    assert model.exists()


def test_delete_refuses_an_active_download(temp_db: None, monkeypatch, capsys, tmp_path) -> None:
    from finetune_studio import db
    from finetune_studio.inference.remote import RemoteEngine
    from finetune_studio.models import hf_cache
    from finetune_studio.webui.routes import hf_models

    monkeypatch.setattr(hf_models, "_LOCAL", tmp_path)
    monkeypatch.setattr(hf_cache, "_LOCAL", tmp_path)
    (tmp_path / "Org__Tiny").mkdir()
    monkeypatch.setattr(db, "list_hf_downloads_in_progress", lambda limit: [{"repo_id": "Org/Tiny"}])
    monkeypatch.setattr(RemoteEngine, "_status", lambda _self, fresh=False: {"loaded": False})
    code, stdout, stderr = run_cli(monkeypatch, capsys, "models", "delete", "Org/Tiny", "--yes")
    assert code == 1 and not stdout and "download is still running" in stderr
    assert (tmp_path / "Org__Tiny").exists()


def test_invalid_download_ids_are_reported_without_tracebacks(monkeypatch, capsys, temp_db: None) -> None:
    code, stdout, stderr = run_cli(monkeypatch, capsys, "models", "download", "../outside")
    assert code == 1 and not stdout and "invalid repo_id" in stderr
    code, stdout, stderr = run_cli(monkeypatch, capsys, "models", "delete", "../outside", "--yes")
    assert code == 1 and not stdout and "invalid repo_id" in stderr


def test_models_help_lists_library_actions(monkeypatch, capsys) -> None:
    code, stdout, stderr = run_cli(monkeypatch, capsys, "models", "--help")
    assert code == 0 and not stderr
    assert all(action in stdout for action in ("search", "local", "download", "download-status", "delete"))
