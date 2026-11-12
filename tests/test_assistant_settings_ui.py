"""Settings -> AI assistant drives every assistant / connection endpoint (UI ships with the API)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui"
TEMPLATE = (ROOT / "templates" / "settings.html").read_text(encoding="utf-8")
SCRIPT = (ROOT / "static" / "js" / "settings.js").read_text(encoding="utf-8")


def test_the_card_has_every_control() -> None:
    for el in ("assistant-card", "assistant-main", "btn-assistant-main", "connections-table", "btn-conn-add", "conn-form",
               "conn-name", "conn-preset", "conn-url", "conn-key", "conn-model", "btn-conn-models", "conn-effort",
               "conn-concurrency", "btn-conn-save", "btn-conn-test", "btn-conn-clear-key", "local-select",
               "btn-local-use", "btn-local-rec", "btn-local-test", "assistant-jobs", "vision-local-model",
               "btn-vision-test", "assistant-status", "judge-auto"):
        assert f'id="{el}"' in TEMPLATE, el
    assert 'type="password"' in TEMPLATE.split('id="conn-key"', 1)[1][:200] or 'id="conn-key" type="password"' in TEMPLATE
    # The old cards and their removed endpoints are gone.
    for gone in ('id="helper-card"', 'id="vision-card"', 'id="judge-provider"'):
        assert gone not in TEMPLATE, gone


def test_the_script_calls_every_endpoint() -> None:
    for call in ("'GET', '/api/settings/assistant'", "'PUT', '/api/settings/assistant'",
                 "'/api/settings/assistant/jobs/'", "'POST', '/api/settings/connections'",
                 "'PUT', '/api/settings/connections/'", "'DELETE', '/api/settings/connections/'",
                 "'/api/settings/connections/test'", "'/api/settings/connections/models'",
                 "'/api/settings/helper/local'", "'/api/settings/helper/recommended/download'",
                 "'/api/settings/helper/test'", "'/api/settings/vision/test'"):
        assert call in SCRIPT, call
    for gone in ("'/api/settings/vision', 'PUT'", "'/api/settings/helper', 'PUT'", "/api/settings/helper/models"):
        assert gone not in SCRIPT, gone
    assert "instructions:" in SCRIPT and "data-job-instr" in SCRIPT     # per-job instructions are saved
    assert "api_key" in SCRIPT and "conn-key').value = ''" in SCRIPT    # the key field is write-only


def test_the_settings_page_renders_the_card(client) -> None:
    body = client.get("/settings").text
    assert 'id="assistant-card"' in body and "/static/js/settings.js?v=" in body
