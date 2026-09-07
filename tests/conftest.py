"""Shared test fixtures for finetune-studio tests.

Provides:
- temp_db: real SQLite database in tmp dir + patched settings
- mock_settings: FakeSettings with patched db_path for app code
- client: FastAPI TestClient bound to the webui app with patched DB
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
    """Create a temp SQLite DB, patch settings.db_path to point at it."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)  # let init_db create it

    # Patch the settings module's db_path
    from finetune_studio import config as cfg

    class FakeSettings:
        db_path = path
        host = "127.0.0.1"
        port = 7860

    # Replace the singleton
    monkeypatch.setattr(cfg, "settings", FakeSettings())

    # Patch any module that captured the old settings reference
    try:
        from finetune_studio.db import connection
        monkeypatch.setattr(connection, "settings", FakeSettings())
    except Exception:
        pass

    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def mock_settings(temp_db, monkeypatch):
    """Provide the FakeSettings (depends on temp_db so db_path is real)."""
    from finetune_studio import config as cfg

    class FakeSettings:
        db_path = temp_db
        host = "127.0.0.1"
        port = 7860

    # Make `from finetune_studio.config import settings` work everywhere
    monkeypatch.setattr(cfg, "settings", FakeSettings())
    return FakeSettings()


@pytest.fixture
def client(mock_settings, monkeypatch):
    """FastAPI TestClient for the webui app."""
    # Patch HF Hub calls so tests don't hit the network
    from fastapi.testclient import TestClient

    # Defer import until after settings patched
    try:
        from finetune_studio.webui.app import app
    except Exception:
        try:
            from finetune_studio.webui import app as app_mod
            app = app_mod.app
        except Exception:
            pytest.skip("webui app not importable")

    return TestClient(app)
