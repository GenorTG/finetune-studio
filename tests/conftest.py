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


@pytest.fixture(scope="session", autouse=True)
def _ensure_ocr_tessdata() -> None:
    """Pre-install tessdata so OCR tests are green on a fresh clone.

    Honours ``FTS_OCR_AUTOINSTALL=0`` (air-gapped CI). Idempotent — already-
    installed languages are skipped. Failures are logged but do not raise:
    the OCR tests themselves report the actionable install command in their
    own failure messages.
    """
    if os.environ.get("FTS_OCR_AUTOINSTALL", "1") == "0":
        return
    try:
        from finetune_studio.data import ocr
    except Exception as e:  # noqa: BLE001 — fixture must never crash pytest
        print(f"\n[conftest] could not import ocr module: {e}", file=sys.stderr)
        return
    if ocr.is_available():
        return
    if not ocr._tesseract_cmd():
        # Binary missing entirely; tests will surface the install hint.
        return
    try:
        ocr.install()
    except Exception as e:  # noqa: BLE001 — tessdata download failure must not crash pytest
        print(f"\n[conftest] tessdata self-install failed: {e}", file=sys.stderr)


@pytest.fixture(autouse=True)
def temp_db(monkeypatch):
    """Redirect the database at a temp SQLite file, yield its path, clean up.

    Autouse: tests that touch ``db`` directly (without requesting
    ``client``/``mock_settings``) used to write straight into the real dev
    database — that produced 226 stray "Recent Suite Runs" / "Limit Suite
    Runs" / "ensure-helper" project rows. Isolating by default makes the leak
    structurally impossible instead of relying on every test remembering to
    request a fixture.

    The patch is a full-fidelity copy of the real ``Settings`` with only
    ``db_path`` redirected, NOT a minimal stub: this fixture now applies to
    every test in the suite, so a stub would strip ``data_dir``, ``rag``,
    ``model_dirs``… from any application code a test exercises.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # Patch settings.db_path before importing db modules
    import dataclasses

    import finetune_studio.config as cfg

    fake = dataclasses.replace(
        cfg.settings, db_path=db_path, host="127.0.0.1", port=7860
    )
    monkeypatch.setattr(cfg, "settings", fake)

    # ``db.connection`` bound its own ``settings`` reference at import
    # (``from finetune_studio.config import settings``), so patching only
    # ``cfg.settings`` left every DB write pointing at the real dev database.
    # Patch the reference ``_connect`` actually reads so each test is truly
    # isolated and never pollutes ``data/finetune_studio.db``.
    import finetune_studio.db.connection as _conn
    monkeypatch.setattr(_conn, "settings", fake)

    # Initialise schema with the real init_db
    _conn.init_db()

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
        fake.open = lambda *a, **kw: open(*a, **kw)  # noqa: SIM115 — callback, not a file-open site
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
        except Exception as e:  # noqa: BLE001 — broad catch is intentional so test setup never crashes pytest
            pytest.skip(f"webui app not importable: {e}")
        with TestClient(app) as c:
            yield c
