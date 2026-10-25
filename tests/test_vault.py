"""API keys are stored encrypted (``access.vault``): never plain text in the provider DB, never returned."""
from __future__ import annotations

import sqlite3
import stat

import pytest

from finetune_studio.access import vault


def _raw_key(pid: str) -> str:
    import finetune_studio.models.manager as mgr_mod
    with sqlite3.connect(mgr_mod._DB_PATH) as c:
        return c.execute("SELECT api_key FROM model_providers WHERE id = ?", (pid,)).fetchone()[0]


def test_seal_roundtrip_and_owner_only_key_file():
    sealed = vault.seal("sk-secret-123")
    assert sealed.startswith(vault.PREFIX) and "sk-secret-123" not in sealed
    assert vault.unseal(sealed) == "sk-secret-123"
    assert vault.seal("") == "" and vault.unseal("") == ""
    assert vault.seal(sealed) == sealed  # sealing twice is a no-op
    assert stat.S_IMODE(vault.key_file().stat().st_mode) == 0o600
    assert stat.S_IMODE(vault.key_file().parent.stat().st_mode) == 0o700


def test_provider_key_is_encrypted_in_db_and_never_listed():
    from finetune_studio.models.manager import ModelManager, get_manager

    mgr = get_manager()
    mgr.upsert_provider(id="api-x", name="X", kind="openai_compat", model_id="m", base_url="https://x", api_key="sk-plain-1")
    assert "sk-plain-1" not in _raw_key("api-x")
    assert ModelManager._api_key_for("api-x") == "sk-plain-1"
    row = mgr.get_provider("api-x")
    assert row["api_key_set"] is True and row["api_key_state"] == "set"
    assert "sk-plain-1" not in repr(mgr.list_providers())


def test_legacy_plaintext_key_is_sealed_on_startup():
    import finetune_studio.models.manager as mgr_mod

    mgr_mod.get_manager()
    with sqlite3.connect(mgr_mod._DB_PATH) as c:
        c.execute("INSERT INTO model_providers (id, name, kind, model_id, base_url, api_key, extra_json, created_at) "
                  "VALUES ('api-old', 'Old', 'openai_compat', 'm', 'https://x', 'sk-legacy-9', '{}', 0)")
    mgr_mod._ensure_db()
    assert vault.is_sealed(_raw_key("api-old"))
    assert mgr_mod.ModelManager._api_key_for("api-old") == "sk-legacy-9"
    assert stat.S_IMODE(mgr_mod._DB_PATH.stat().st_mode) == 0o600


def test_lost_vault_key_reports_unreadable_not_a_wrong_key():
    from finetune_studio.models.manager import ModelManager, get_manager

    mgr = get_manager()
    mgr.upsert_provider(id="api-y", name="Y", kind="openai_compat", model_id="m", base_url="https://y", api_key="sk-y")
    vault.key_file().unlink()
    row = mgr.get_provider("api-y")
    assert row["api_key_state"] == "unreadable" and row["api_key_set"] is False
    assert ModelManager._api_key_for("api-y") == ""
    with pytest.raises(vault.VaultError):
        vault.unseal(_raw_key("api-y"))


def test_helper_settings_never_return_the_key():
    from fastapi.testclient import TestClient

    from finetune_studio.webui.app import app

    c = TestClient(app)
    r = c.put("/api/settings/helper", json={"seat": "local", "base_url": "https://api.example.com/v1", "model": "m",
                                            "api_key": "sk-never-shown"})
    assert r.status_code == 200, r.text
    body = c.get("/api/settings/helper").text
    assert "sk-never-shown" not in body
    assert c.get("/api/settings/helper").json()["api"]["key_state"] == "set"
    assert "sk-never-shown" not in _raw_key("api-helper")
