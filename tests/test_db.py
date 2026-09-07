"""Tests for db/ package — projects, rags, runs, benchmarks."""
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_db(monkeypatch):
    """Override the DB path with a temp file, re-init, and clean up."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    # Patch settings so init_db uses our temp path
    with patch("finetune_studio.config.settings") as mock_settings:
        mock_settings.db_path = db_path
        # Re-init the module with the patched settings
        import importlib, finetune_studio.db, finetune_studio.db.connection
        monkeypatch.setattr(finetune_studio.db.connection, "_settings", mock_settings)
        # Re-create the schema
        import sqlite3
        conn = sqlite3.connect(db_path)
        with open(os.path.join(os.path.dirname(finetune_studio.db.connection.__file__), "connection.py")) as f:
            src = f.read()
        # Just use raw SQL to set up schema
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
                base_model TEXT NOT NULL DEFAULT '', system_prompt TEXT NOT NULL DEFAULT '',
                production_run TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS project_rags (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '',
                store_path TEXT NOT NULL, doc_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE);
            CREATE INDEX IF NOT EXISTS idx_rags_project ON project_rags(project_id);
            CREATE TABLE IF NOT EXISTS training_runs (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, name TEXT NOT NULL,
                config_json TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'pending',
                started_at REAL, finished_at REAL, created_at REAL NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, name TEXT NOT NULL,
                score REAL, details TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS data_reviews (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, run_id TEXT,
                file_hash TEXT NOT NULL, decision TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
                reviewed_at REAL NOT NULL);
        """)
        conn.commit()
        conn.close()
        yield db_path
    os.unlink(db_path)


@pytest.fixture
def mock_settings(temp_db, monkeypatch):
    """Patch settings so the whole app uses the temp DB."""
    import finetune_studio.config as cfg
    m = MagicMock()
    m.db_path = temp_db
    m.rag = MagicMock()
    m.rag.store_path = "data/rag_store"
    m.rag.documents_path = "data/rag_documents"
    m.rag.embedding_model = "all-MiniLM-L6-v2"
    m.rag.min_score = 0.3
    m.rag.chunk_size = 512
    m.rag.chunk_overlap = 50
    m.projects_dir = "data/projects"
    m.shared_models_dir = ".finetune-studio/shared_models"
    m.hf_cache_dir = ".finetune-studio/hf_models"
    m.data_dir = "data"
    monkeypatch.setattr(cfg, "settings", m)
    return m


# ── Projects ─────────────────────────────────────────────────────────────────

class TestProjects:
    def test_create_and_get_project(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Test Project", description="A test")
        assert pid is not None
        p = db.get_project(pid)
        assert p is not None
        assert p["name"] == "Test Project"
        assert p["description"] == "A test"

    def test_create_project_with_fields(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(
            name="My Project",
            description="Description here",
            base_model="Qwen/Qwen2-7B",
            system_prompt="You are helpful.",
        )
        p = db.get_project(pid)
        assert p["base_model"] == "Qwen/Qwen2-7B"
        assert p["system_prompt"] == "You are helpful."

    def test_update_project(self, mock_settings):
        import finetune_studio.db as db
        import time
        pid = db.create_project(name="Before", description="old")
        db.update_project(pid, name="After", description="new")
        p = db.get_project(pid)
        assert p["name"] == "After"
        assert p["description"] == "new"

    def test_delete_project(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="To Delete", description="")
        db.delete_project(pid)
        assert db.get_project(pid) is None

    def test_list_projects(self, mock_settings):
        import finetune_studio.db as db
        id1 = db.create_project(name="Project A", description="")
        id2 = db.create_project(name="Project B", description="")
        projects = db.list_projects()
        names = {p["name"] for p in projects}
        assert "Project A" in names
        assert "Project B" in names

    def test_get_nonexistent_project(self, mock_settings):
        import finetune_studio.db as db
        assert db.get_project("does_not_exist") is None


# ── RAGs ────────────────────────────────────────────────────────────────────

class TestRags:
    def test_create_and_get_rag(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="RAG Test", description="")
        rid = db.create_rag(
            project_id=pid, name="My RAG", description="A RAG",
            store_path="data/rag_store/my_rag",
        )
        assert rid is not None
        rag = db.get_rag(rid)
        assert rag["name"] == "My RAG"
        assert rag["project_id"] == pid

    def test_update_rag(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="RAG Update", description="")
        rid = db.create_rag(project_id=pid, name="Old Name", store_path="/tmp/r")
        db.update_rag(rid, name="New Name", doc_count=10)
        rag = db.get_rag(rid)
        assert rag["name"] == "New Name"
        assert rag["doc_count"] == 10

    def test_delete_rag(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="RAG Delete", description="")
        rid = db.create_rag(project_id=pid, name="To Delete", store_path="/tmp/r")
        db.delete_rag(rid)
        assert db.get_rag(rid) is None

    def test_list_rags_for_project(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="RAG List", description="")
        r1 = db.create_rag(project_id=pid, name="RAG 1", store_path="/tmp/1")
        r2 = db.create_rag(project_id=pid, name="RAG 2", store_path="/tmp/2")
        rags = db.list_rags(pid)
        names = {r["name"] for r in rags}
        assert names == {"RAG 1", "RAG 2"}


# ── Runs ─────────────────────────────────────────────────────────────────────

class TestRuns:
    def test_create_and_get_run(self, mock_settings):
        import finetune_studio.db as db
        import time
        pid = db.create_project(name="Run Test", description="")
        rid = db.create_run(
            project_id=pid, name="Run 1",
            config_json={"lr": 1e-4, "epochs": 3},
        )
        assert rid is not None
        run = db.get_run(rid)
        assert run["name"] == "Run 1"
        assert run["status"] == "pending"

    def test_update_run_status(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Status Test", description="")
        rid = db.create_run(project_id=pid, name="Run", config_json={})
        db.update_run(rid, status="training", started_at=1234567890.0)
        run = db.get_run(rid)
        assert run["status"] == "training"
        assert run["started_at"] == 1234567890.0

    def test_delete_run(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Delete Run", description="")
        rid = db.create_run(project_id=pid, name="To Delete", config_json={})
        db.delete_run(rid)
        assert db.get_run(rid) is None

    def test_list_runs(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="List Runs", description="")
        r1 = db.create_run(project_id=pid, name="Run 1", config_json={})
        r2 = db.create_run(project_id=pid, name="Run 2", config_json={})
        runs = db.list_runs(pid)
        assert len(runs) >= 2

    def test_list_runs_with_status_filter(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Filter Runs", description="")
        r1 = db.create_run(project_id=pid, name="Pending Run", config_json={})
        r2 = db.create_run(project_id=pid, name="Done Run", config_json={})
        db.update_run(r2, status="done")
        pending = db.list_runs(pid, status="pending")
        done = db.list_runs(pid, status="done")
        pending_names = {r["name"] for r in pending}
        done_names = {r["name"] for r in done}
        assert "Pending Run" in pending_names
        assert "Done Run" in done_names
        assert "Done Run" not in pending_names


# ── Benchmarks ───────────────────────────────────────────────────────────────

class TestBenchmarks:
    def test_create_and_get_benchmark(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Benchmark Test", description="")
        rid = db.create_run(project_id=pid, name="Run", config_json={})
        bid = db.create_benchmark(
            run_id=rid, name="Benchmark 1", score=0.85,
            details='{"loss": 0.15}',
        )
        assert bid is not None
        b = db.get_benchmark(bid)
        assert b["score"] == 0.85
        assert b["run_id"] == rid

    def test_list_benchmarks(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="List Bench", description="")
        rid = db.create_run(project_id=pid, name="Run", config_json={})
        db.create_benchmark(run_id=rid, name="B1", score=0.8)
        db.create_benchmark(run_id=rid, name="B2", score=0.9)
        benchmarks = db.list_benchmarks(rid)
        assert len(benchmarks) == 2


# ── Reviews ─────────────────────────────────────────────────────────────────

class TestReviews:
    def test_record_and_list_review(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Review Test", description="")
        review_id = db.record_review(
            project_id=pid, file_hash="abc123",
            decision="approved", note="Looks good",
        )
        assert review_id is not None
        reviews = db.list_review(pid)
        assert len(reviews) >= 1
        assert reviews[0]["file_hash"] == "abc123"
        assert reviews[0]["decision"] == "approved"
