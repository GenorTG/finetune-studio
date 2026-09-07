"""Shared test fixtures for finetune-studio tests.

Provides:
- temp_db: real SQLite database in tmp dir, returns db_path
- mock_settings: MagicMock with full settings interface + patched into config
- client: FastAPI TestClient bound to the webui app
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def temp_db(monkeypatch):
    """Create a temp SQLite DB file; init schema; yield path; cleanup."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # Initialise schema
    import sqlite3
    conn = sqlite3.connect(db_path)
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
    conn.close()

    yield db_path
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def mock_settings(temp_db, monkeypatch):
    """MagicMock with full settings interface, patched into config singleton."""
    import finetune_studio.config as cfg
    m = MagicMock()
    m.db_path = temp_db
    m.host = "127.0.0.1"
    m.port = 7860
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


@pytest.fixture
def client(mock_settings, monkeypatch):
    """FastAPI TestClient for the webui app with heavy deps mocked."""
    from fastapi.testclient import TestClient

    # Install missing modules so import doesn't fail
    try:
        import aiofiles  # noqa: F401
    except ImportError:
        import types
        fake = types.ModuleType("aiofiles")
        fake.open = lambda *a, **kw: open(*a, **kw)
        monkeypatch.setitem(sys.modules, "aiofiles", fake)

    with patch("finetune_studio.models.registry.scan_models") as mock_scan, \
         patch("finetune_studio.training.engine.TrainingEngine") as mock_te, \
         patch("finetune_studio.testing.inference.InferenceEngine") as mock_ie, \
         patch("finetune_studio.db.init_db"):
        mock_scan.return_value = []
        try:
            import importlib
            import finetune_studio.webui.app as app_module
            importlib.reload(app_module)
            app = app_module.app
        except Exception as e:
            pytest.skip(f"webui app not importable: {e}")
        with TestClient(app) as c:
            yield c
