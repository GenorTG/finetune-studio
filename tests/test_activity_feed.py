"""Activity feed aggregates persisted history, not just live memory.

The global activity drawer used to only show whatever was still resident in
process memory (the training engine's live state, in-flight data-prep runs,
in-flight downloads). Anything that had finished — or ran before a restart —
vanished, and whole subsystems (benchmarks, exports, RAG builds, system
updates) never appeared at all. These tests lock in that ``collect_activity``
reads the durable DB tables so every kind of work shows up.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from typing import Any

import pytest


@pytest.fixture
def iso_db(monkeypatch):
    """A truly isolated SQLite DB for this test.

    The shared ``temp_db`` conftest fixture patches ``config.settings`` but
    ``db.connection`` bound its own ``settings`` reference at import, so writes
    leak into the real dev DB. We patch the reference ``_connect`` actually
    reads (``finetune_studio.db.connection.settings``) so each test gets a
    clean, disposable database and pollutes nothing.
    """
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    import finetune_studio.db.connection as conn

    class _Fake:
        db_path = path
        host = "127.0.0.1"
        port = 7860

    monkeypatch.setattr(conn, "settings", _Fake())
    conn.init_db()
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def _stub_live(monkeypatch):
    """Stub the heavy live-state imports so collect_activity stays fast/clean.

    The live blocks import ``finetune_studio.webui.app`` (torch, models, …).
    For a pure persisted-history test we inject an idle stand-in so those
    blocks contribute nothing and the persisted rows are what we assert on.
    """
    app_mod = types.ModuleType("finetune_studio.webui.app")

    class _State:
        status = "idle"
        current_step = 0
        total_steps = 0
        loss = 0.0
        elapsed = 0
        message = ""

    class _Engine:
        state = _State()
        current_run_id = ""

    class _Infer:
        model = None
        model_path = ""

    app_mod.training_engine = _Engine()
    app_mod.inference_engine = _Infer()
    monkeypatch.setitem(sys.modules, "finetune_studio.webui.app", app_mod)
    yield


def _kinds(payload: dict[str, Any]) -> set[str]:
    return {t["kind"] for t in payload["tasks"] if t.get("kind") != "_error"}


def test_empty_db_has_no_tasks(iso_db, _stub_live):
    from finetune_studio.webui.routes.activity import collect_activity
    payload = collect_activity()
    assert _kinds(payload) == set()
    assert payload["active_count"] == 0


def test_every_persisted_kind_shows(iso_db, _stub_live):
    from finetune_studio import db
    from finetune_studio.webui.routes.activity import collect_activity

    proj = db.create_project("Feed Project")
    pid = proj["id"]

    # training run (finished)
    run = db.create_run(pid, "run-1", base_model="Qwen/Qwen3-0.6B")
    db.update_run(run["id"], status="completed", final_loss=0.123,
                  started_at=1000.0, finished_at=1100.0)

    # benchmark against that run
    db.create_benchmark(run["id"], "held-out", {"weighted": 0.97}, time_ms=800)

    # data-prep run (finished)
    dp = db.create_data_prep_run(pid, filename="notes.pdf")
    db.mark_data_prep_running(dp["id"])
    db.mark_data_prep_done(dp["id"], qa_total=20, qa_approved=18)

    # RAG build (finished)
    rag = db.create_rag(pid, "corpus")
    build = db.create_rag_build(pid, rag["id"])
    db.mark_rag_build_running(build["id"])
    db.mark_rag_build_done(build["id"], doc_count=5, chunk_count=120)

    # HF download (finished)
    job = db.create_hf_download("Qwen/Qwen3-0.6B", "model.safetensors")
    db.mark_hf_download_done(job["id"], path="/models/qwen")

    # model export (finished)
    exp = db.create_export(pid, run["id"], format="gguf", quant="Q8_0")
    db.mark_export_running(exp["id"])
    db.mark_export_done(exp["id"], output_path="/out/model-q8.gguf",
                        size_human="640 MB")

    payload = collect_activity()
    kinds = _kinds(payload)
    for expected in ("training", "benchmark", "data_prep", "rag_ready",
                     "download", "export"):
        assert expected in kinds, f"{expected} missing from feed: {kinds}"

    # project resolution: training/benchmark/data_prep/export carry the pid
    by_kind = {t["kind"]: t for t in payload["tasks"]}
    assert by_kind["training"]["project_id"] == pid
    assert by_kind["benchmark"]["project_id"] == pid
    assert by_kind["data_prep"]["project_id"] == pid
    assert by_kind["export"]["project_id"] == pid

    # finished rows are still present even though nothing is "active"
    assert payload["active_count"] == 0


def test_finished_training_survives_without_live_state(iso_db, _stub_live):
    from finetune_studio import db
    from finetune_studio.webui.routes.activity import collect_activity

    proj = db.create_project("P")
    run = db.create_run(proj["id"], "old-run")
    db.update_run(run["id"], status="completed", finished_at=500.0)

    payload = collect_activity()
    training = [t for t in payload["tasks"] if t["kind"] == "training"]
    assert len(training) == 1
    assert training[0]["run_id"] == run["id"]
    assert training[0]["status"] == "done"


def test_live_training_dedups_with_persisted_row(iso_db, monkeypatch):
    """A running engine + its DB row collapse to ONE training task."""
    from finetune_studio import db

    proj = db.create_project("Live P")
    pid = proj["id"]
    run = db.create_run(pid, "live-run")
    run_id = f"{pid}-{run['id']}"  # app uses composite current_run_id
    db.update_run(run["id"], status="training", started_at=10.0)

    # Live engine reporting the same run mid-flight.
    app_mod = types.ModuleType("finetune_studio.webui.app")

    class _State:
        status = "training"
        current_step = 5
        total_steps = 10
        loss = 0.5
        elapsed = 30
        message = "training"

    class _Engine:
        state = _State()
        current_run_id = run_id

    class _Infer:
        model = None
        model_path = ""

    app_mod.training_engine = _Engine()
    app_mod.inference_engine = _Infer()
    monkeypatch.setitem(sys.modules, "finetune_studio.webui.app", app_mod)

    from finetune_studio.webui.routes.activity import collect_activity
    payload = collect_activity()
    training = [t for t in payload["tasks"] if t["kind"] == "training"]
    assert len(training) == 1, training
    # The live row (with step progress) wins.
    assert "step 5/10" in training[0]["message"]


def test_deleted_project_rows_are_skipped(iso_db, _stub_live):
    from finetune_studio import db
    from finetune_studio.webui.routes.activity import collect_activity

    proj = db.create_project("Doomed")
    pid = proj["id"]
    run = db.create_run(pid, "r")
    db.update_run(run["id"], status="completed", finished_at=1.0)
    db.delete_project(pid)

    payload = collect_activity()
    assert _kinds(payload) == set()
