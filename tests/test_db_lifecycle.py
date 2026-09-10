"""Lifecycle tests: assert every task type populates its DB fields end-to-end.

Each test simulates the full lifecycle for one task type (training,
benchmark, data-prep, rag-ingest, hf-download) and asserts that every
field the UI / activity feed / dashboard expects is actually present
in the persisted row.

Uses the real SQLite DB via the `mock_settings` fixture — same DB
schema, same CRUD helpers, no mocks.
"""

from __future__ import annotations

import os
import time
import tempfile

import pytest


# ── Training run lifecycle ───────────────────────────────────────────────


class TestTrainingRunLifecycle:
    """A training run should land these fields on every state transition:
    status, started_at (first active transition), finished_at (done/error),
    output_path (the parent <output_dir>/), error (on failure), metrics.
    """

    def test_run_created_with_default_fields(self, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="Lifecycle Train", description="")["id"]
        rid = db.create_run(project_id=pid, name="Train Run",
                            base_model="Qwen/Qwen2-7B",
                            settings_obj={"lora_rank": 64})["id"]
        run = db.get_run(rid)
        assert run["status"] == "created"
        assert run["started_at"] is None
        assert run["finished_at"] is None
        assert run["output_path"] == ""
        assert run["error"] == ""

    def test_running_transition_writes_started_at(self, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="T", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m", settings_obj={})["id"]
        before = time.time()
        db.update_run(rid, status="running", started_at=time.time())
        run = db.get_run(rid)
        assert run["status"] == "running"
        assert run["started_at"] is not None
        assert run["started_at"] >= before

    def test_done_transition_writes_finished_at_and_output_path(self, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="T", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m", settings_obj={})["id"]
        with tempfile.TemporaryDirectory() as out:
            os.makedirs(os.path.join(out, "adapter"))
            db.update_run(rid, status="running", started_at=time.time())
            db.update_run(rid, status="done", finished_at=time.time(),
                          output_path=out)
            run = db.get_run(rid)
            assert run["status"] == "done"
            assert run["finished_at"] is not None
            assert run["output_path"] == out
            # Adapter dir should be findable at output_path/adapter
            assert os.path.isdir(os.path.join(run["output_path"], "adapter"))

    def test_error_transition_writes_error_and_output_path_if_adapter_exists(
        self, mock_settings
    ):
        from finetune_studio import db
        pid = db.create_project(name="T", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m", settings_obj={})["id"]
        with tempfile.TemporaryDirectory() as out:
            os.makedirs(os.path.join(out, "adapter"))
            db.update_run(rid, status="running", started_at=time.time())
            db.update_run(rid, status="error", finished_at=time.time(),
                          error="CUDA OOM after step 100", output_path=out)
            run = db.get_run(rid)
            assert run["status"] == "error"
            assert run["error"] == "CUDA OOM after step 100"
            # output_path should still be populated so user can recover the adapter
            assert run["output_path"] == out

    def test_error_transition_omits_output_path_without_adapter(self, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="T", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m", settings_obj={})["id"]
        db.update_run(rid, status="running", started_at=time.time())
        db.update_run(rid, status="error", finished_at=time.time(),
                      error="load failed before any save")
        run = db.get_run(rid)
        assert run["status"] == "error"
        assert run["error"] == "load failed before any save"
        assert run["output_path"] == ""

    def test_metrics_and_settings_round_trip(self, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="T", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m",
                            settings_obj={"lora_rank": 32, "lr": 1e-4})["id"]
        db.update_run(rid, metrics_json={"step": 100, "loss": 1.23, "epoch": 0.5})
        run = db.get_run(rid)
        assert run["metrics"]["step"] == 100
        assert run["metrics"]["loss"] == 1.23
        assert run["settings"]["lora_rank"] == 32

    def test_error_field_in_allowed_set(self, mock_settings):
        """Regression: the engine sets .error via update_run — make sure
        the allowed-set whitelist actually contains 'error'."""
        from finetune_studio import db
        pid = db.create_project(name="T", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m", settings_obj={})["id"]
        # This should NOT raise or silently drop the field.
        db.update_run(rid, error="some failure")
        assert db.get_run(rid)["error"] == "some failure"


# ── Benchmark lifecycle ──────────────────────────────────────────────────


class TestBenchmarkLifecycle:
    """benchmark_runs: ran_at, time_ms, scores_json, suite_name. One row per
    successful benchmark. Failure case still returns 200 with {error}.
    """

    def test_create_benchmark_writes_required_fields(self, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="B", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m", settings_obj={})["id"]
        before = time.time()
        bid = db.create_benchmark(run_id=rid, suite_name="default",
                                  scores={"loss": 0.5, "perplexity": 12.3},
                                  time_ms=4321)["id"]
        b = db.get_benchmark(bid)
        assert b["suite_name"] == "default"
        assert b["scores"]["loss"] == 0.5
        assert b["scores"]["perplexity"] == 12.3
        assert b["time_ms"] == 4321
        assert b["ran_at"] >= before

    def test_list_benchmarks_for_run(self, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="B", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m", settings_obj={})["id"]
        db.create_benchmark(run_id=rid, suite_name="suite-1", scores={"acc": 0.9})
        db.create_benchmark(run_id=rid, suite_name="suite-2", scores={"acc": 0.7})
        benches = db.list_benchmarks(run_id=rid)
        assert len(benches) == 2
        suites = {b["suite_name"] for b in benches}
        assert suites == {"suite-1", "suite-2"}


# ── Data-prep lifecycle ──────────────────────────────────────────────────


class TestDataPrepLifecycle:
    """data_prep_runs: status, started_at, finished_at, duration_ms,
    qa_total, qa_approved, output_path, error.
    """

    def test_full_lifecycle_done(self, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="P", description="")["id"]
        row = db.create_data_prep_run(
            project_id=pid, filename="doc.pdf", byte_count=12345,
        )
        rid = row["id"]
        assert row["status"] == "queued"
        # mark_running
        db.mark_data_prep_running(rid)
        r = db.get_data_prep_run(rid)
        assert r["status"] == "running"
        assert r["started_at"] is not None
        # simulate some progress (small sleep so duration_ms > 0)
        time.sleep(0.01)
        # mark_done
        db.mark_data_prep_done(rid, qa_total=42, qa_approved=10,
                               output_path="/tmp/qa/pairs.json")
        r = db.get_data_prep_run(rid)
        assert r["status"] == "done"
        assert r["finished_at"] is not None
        assert r["duration_ms"] is not None and r["duration_ms"] > 0
        assert r["qa_total"] == 42
        assert r["qa_approved"] == 10
        assert r["output_path"] == "/tmp/qa/pairs.json"

    def test_lifecycle_failure(self, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="P", description="")["id"]
        rid = db.create_data_prep_run(
            project_id=pid, filename="bad.pdf", byte_count=0,
        )["id"]
        db.mark_data_prep_running(rid)
        db.mark_data_prep_failed(rid, "parser error: unsupported format")
        r = db.get_data_prep_run(rid)
        assert r["status"] == "error"
        assert "parser error" in r["error"]
        assert r["finished_at"] is not None

    def test_list_for_project_returns_recent(self, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="P", description="")["id"]
        for i in range(3):
            db.create_data_prep_run(project_id=pid, filename=f"f{i}.pdf",
                                    byte_count=100 * i)
        runs = db.list_data_prep_for_project(pid)
        assert len(runs) == 3


# ── RAG ingest lifecycle ─────────────────────────────────────────────────


class TestRagIngestLifecycle:
    """project_rags gains: status, last_build_at, last_build_status, error.
    rag_corpora table: one row per build attempt with status, timing,
    final doc/chunk counts.
    """

    def test_rag_corpora_lifecycle_done(self, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="R", description="")["id"]
        rid = db.create_rag(project_id=pid, name="My RAG",
                            store_path="/tmp/rag-test")["id"]
        # Start a build
        build = db.create_rag_build(project_id=pid, rag_id=rid)
        bid = build["id"]
        assert build["status"] == "queued"
        # Mark running
        db.mark_rag_build_running(bid)
        b = db.get_rag_build(bid)
        assert b["status"] == "running"
        assert b["started_at"] is not None
        # Mark done
        time.sleep(0.01)
        db.mark_rag_build_done(bid, doc_count=5, chunk_count=137)
        b = db.get_rag_build(bid)
        assert b["status"] == "done"
        assert b["doc_count"] == 5
        assert b["chunk_count"] == 137
        assert b["finished_at"] is not None
        assert b["duration_ms"] is not None and b["duration_ms"] > 0

    def test_project_rags_widened_status_fields(self, mock_settings):
        """The widened update_rag allowed set must accept the new columns."""
        from finetune_studio import db
        pid = db.create_project(name="R", description="")["id"]
        rid = db.create_rag(project_id=pid, name="X",
                            store_path="/tmp/x")["id"]
        db.update_rag(rid, status="building",
                      last_build_at=time.time(),
                      last_build_status="running")
        rag = db.get_rag(rid)
        assert rag["status"] == "building"
        assert rag["last_build_status"] == "running"
        assert rag["last_build_at"] is not None
        # Update through to 'ready'
        db.update_rag(rid, status="ready",
                      last_build_status="ok",
                      doc_count=10, chunk_count=200)
        rag = db.get_rag(rid)
        assert rag["status"] == "ready"
        assert rag["last_build_status"] == "ok"
        assert rag["doc_count"] == 10
        assert rag["chunk_count"] == 200

    def test_failure_records_error(self, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="R", description="")["id"]
        rid = db.create_rag(project_id=pid, name="X",
                            store_path="/tmp/x")["id"]
        build = db.create_rag_build(project_id=pid, rag_id=rid)
        db.mark_rag_build_running(build["id"])
        db.mark_rag_build_failed(build["id"], "embedding OOM")
        db.update_rag(rid, status="error", last_build_status="failed",
                      error="embedding OOM")
        rag = db.get_rag(rid)
        assert rag["status"] == "error"
        assert rag["last_build_status"] == "failed"
        assert "embedding OOM" in rag["error"]
        # History row
        latest = db.latest_rag_build(rid)
        assert latest["status"] == "error"
        assert "embedding OOM" in latest["error"]


# ── HF download lifecycle ────────────────────────────────────────────────


class TestHfDownloadLifecycle:
    """hf_downloads: status, started_at, finished_at, duration_ms,
    bytes_total, bytes_done, path, error.
    """

    def test_full_lifecycle_completed(self, mock_settings):
        from finetune_studio import db
        job = db.create_hf_download(repo_id="Qwen/Qwen2-1.5B",
                                    filename="model.safetensors")
        jid = job["id"]
        assert job["status"] == "queued"
        db.mark_hf_download_running(jid)
        j = db.get_hf_download(jid)
        assert j["status"] == "downloading"
        assert j["started_at"] is not None
        time.sleep(0.01)
        db.mark_hf_download_done(jid, path="/tmp/Qwen/Qwen2-1.5B/model.safetensors",
                                 bytes_total=2_500_000_000, bytes_done=2_500_000_000)
        j = db.get_hf_download(jid)
        assert j["status"] == "completed"
        assert j["finished_at"] is not None
        assert j["duration_ms"] is not None and j["duration_ms"] > 0
        assert j["bytes_total"] == 2_500_000_000
        assert j["bytes_done"] == 2_500_000_000
        assert j["path"].endswith("model.safetensors")

    def test_failure_records_error(self, mock_settings):
        from finetune_studio import db
        jid = db.create_hf_download(repo_id="bad/repo")["id"]
        db.mark_hf_download_running(jid)
        db.mark_hf_download_failed(jid, "404 not found")
        j = db.get_hf_download(jid)
        assert j["status"] == "error"
        assert j["error"] == "404 not found"
        assert j["finished_at"] is not None

    def test_cancellation(self, mock_settings):
        from finetune_studio import db
        jid = db.create_hf_download(repo_id="cancelled/repo")["id"]
        db.mark_hf_download_running(jid)
        db.mark_hf_download_cancelled(jid)
        j = db.get_hf_download(jid)
        assert j["status"] == "cancelled"
        assert "cancelled" in j["error"].lower()

    def test_list_in_progress_finds_running_and_queued(self, mock_settings):
        from finetune_studio import db
        # queued
        db.create_hf_download(repo_id="queued/r1")
        # running
        jid = db.create_hf_download(repo_id="running/r2")["id"]
        db.mark_hf_download_running(jid)
        # completed — should NOT appear
        jid_done = db.create_hf_download(repo_id="done/r3")["id"]
        db.mark_hf_download_running(jid_done)
        db.mark_hf_download_done(jid_done, path="/tmp/d", bytes_total=1, bytes_done=1)
        # failed — should NOT appear
        in_prog = db.list_hf_downloads_in_progress()
        repos = {j["repo_id"] for j in in_prog}
        assert "queued/r1" in repos
        assert "running/r2" in repos
        assert "done/r3" not in repos

    def test_list_recent_ordered_newest_first(self, mock_settings):
        from finetune_studio import db
        for i in range(3):
            db.create_hf_download(repo_id=f"r{i}")
            time.sleep(0.005)
        recent = db.list_hf_downloads_recent(limit=2)
        assert len(recent) == 2
        # Newest first
        assert recent[0]["created_at"] >= recent[1]["created_at"]


# ── FK + cascade ─────────────────────────────────────────────────────────


class TestForeignKeys:
    """PRAGMA defer_foreign_keys = ON — child rows can be INSERTed in the
    same transaction as the parent without the parent being 'visible'
    to the FK check until COMMIT. And ON DELETE CASCADE cleans up properly.
    """

    def test_cascade_delete_project_removes_runs_and_rags(self, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="C", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m", settings_obj={})["id"]
        rag_id = db.create_rag(project_id=pid, name="rag",
                               store_path="/tmp/r")["id"]
        db.create_data_prep_run(project_id=pid, filename="f", byte_count=1)
        # Delete the project — everything should cascade.
        db.delete_project(pid)
        assert db.get_run(rid) is None
        assert db.get_rag(rag_id) is None
        assert db.list_data_prep_for_project(pid) == []

    def test_cascade_delete_rag_removes_rag_corpora(self, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="C", description="")["id"]
        rid = db.create_rag(project_id=pid, name="rag",
                            store_path="/tmp/r")["id"]
        build_id = db.create_rag_build(project_id=pid, rag_id=rid)["id"]
        # Delete the rag — the build history should cascade.
        db.delete_rag(rid)
        assert db.get_rag_build(build_id) is None


# ── Engine helpers ───────────────────────────────────────────────────────


class TestEngineHelpers:
    """merge_adapter_for_run + engine helpers — no actual model load."""

    def test_merge_skips_when_merged_dir_exists(self, mock_settings):
        from finetune_studio.training.engine import merge_adapter_for_run
        import tempfile
        with tempfile.TemporaryDirectory() as out:
            os.makedirs(os.path.join(out, "adapter"))
            os.makedirs(os.path.join(out, "merged"))
            # Drop a fake file in merged/ so the size is non-zero.
            with open(os.path.join(out, "merged", "model.safetensors"), "wb") as f:
                f.write(b"x" * 100)
            r = merge_adapter_for_run(
                {"output_path": out, "base_model": "/anywhere"}, force=False,
            )
            assert r["skipped"] is True
            assert r["size_bytes"] == 100
            assert r["merged_path"] == os.path.join(out, "merged")

    def test_merge_raises_without_output_path(self, mock_settings):
        from finetune_studio.training.engine import merge_adapter_for_run
        with pytest.raises(ValueError, match="no output_path"):
            merge_adapter_for_run({})

    def test_merge_raises_without_base_model(self, mock_settings):
        from finetune_studio.training.engine import merge_adapter_for_run
        with pytest.raises(ValueError, match="no base_model"):
            merge_adapter_for_run({"output_path": "/tmp/x"})

    def test_merge_raises_without_adapter_dir(self, mock_settings):
        from finetune_studio.training.engine import merge_adapter_for_run
        with tempfile.TemporaryDirectory() as out:
            with pytest.raises(ValueError, match="adapter dir not found"):
                merge_adapter_for_run(
                    {"output_path": out, "base_model": "/anywhere"},
                )

    def test_merge_skip_via_env_var(self, mock_settings, monkeypatch):
        """FTS_SKIP_MERGE=1 short-circuits the actual merge and writes a
        marker file so a follow-up test can confirm the merge was attempted."""
        from finetune_studio.training.engine import merge_adapter_for_run
        monkeypatch.setenv("FTS_SKIP_MERGE", "1")
        with tempfile.TemporaryDirectory() as out:
            os.makedirs(os.path.join(out, "adapter"))
            r = merge_adapter_for_run(
                {"output_path": out, "base_model": "/anywhere"}, force=True,
            )
            assert r["skipped"] is True
            assert os.path.exists(os.path.join(out, "merged", "SKIPPED_BY_TEST"))

    def test_engine_helpers(self, mock_settings):
        from finetune_studio.training.engine import (
            TrainingConfig, _dir_size, _human_size,
        )
        cfg = TrainingConfig()
        assert cfg.merge_on_save is False
        assert _human_size(0) == "0 B"
        assert "KB" in _human_size(2048)
        assert "MB" in _human_size(2 * 1024 * 1024)
        assert "GB" in _human_size(2 * 1024 ** 3)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a"), "wb") as f:
                f.write(b"x" * 100)
            assert _dir_size(d) == 100
        # Merging on save still possible via constructor kwarg
        cfg2 = TrainingConfig(merge_on_save=True)
        assert cfg2.merge_on_save is True
