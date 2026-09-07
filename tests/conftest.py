"""Shared test fixtures for finetune-studio tests.

Provides:
- temp_db: real SQLite database in tmp dir, schema initialized via init_db()
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
    """Create a temp SQLite DB file, init schema, yield path, cleanup."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # Patch settings.db_path before importing db modules
    import finetune_studio.config as cfg

    class _Fake:
        pass
    _Fake.db_path = db_path
    _Fake.host = "127.0.0.1"
    _Fake.port = 7860
    monkeypatch.setattr(cfg, "settings", _Fake())

    # Initialise schema with the real init_db
    from finetune_studio.db.connection import init_db
    init_db()

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

    # Stub missing aiofiles
    try:
        import aiofiles  # noqa: F401
    except ImportError:
        import types
        fake = types.ModuleType("aiofiles")
        fake.open = lambda *a, **kw: open(*a, **kw)
        monkeypatch.setitem(sys.modules, "aiofiles", fake)

    with patch("finetune_studio.models.registry.scan_models") as mock_scan, \
         patch("finetune_studio.training.engine.TrainingEngine"), \
         patch("finetune_studio.testing.inference.InferenceEngine"):
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
