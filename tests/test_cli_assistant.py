"""CLI parity for Settings → AI assistant connections and job routing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from finetune_studio.cli import _registry
from finetune_studio.models import assistant, helper
from finetune_studio.models.manager import get_manager


def run_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
            *argv: str) -> tuple[int, str, str]:
    capsys.readouterr()  # discard app-lifespan model scan output before asserting command output
    monkeypatch.setattr(sys, "argv", ["fts", *argv])
    try:
        _registry.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    else:
        code = 0
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def add_connection(*, api_key: str = "key-not-for-output", provider_id: str = "api-cli-test") -> str:
    get_manager().upsert_provider(id=provider_id, name="CLI test", kind="openai_compat", model_id="test-model",
                                 base_url="https://api.example/v1", api_key=api_key)
    return provider_id


def test_show_and_connection_list_never_expose_saved_keys(monkeypatch, capsys, client) -> None:
    secret = "do-not-print-this-key"
    pid = add_connection(api_key=secret)
    assistant.set_assistant_provider_id(helper.DEFAULT_HELPER_PROVIDER_ID)

    code, out, err = run_cli(monkeypatch, capsys, "assistant", "show", "--json")
    assert code == 0 and not err
    state = json.loads(out)
    assert state["main"]["id"] == helper.DEFAULT_HELPER_PROVIDER_ID
    assert any(row["id"] == pid and row["key_set"] for row in state["connections"])
    assert any(row["id"] == "judge" for row in state["jobs"])
    assert secret not in out

    code, out, err = run_cli(monkeypatch, capsys, "assistant", "connections", "list", "--json")
    assert code == 0 and not err
    listed = json.loads(out)["connections"]
    assert next(row for row in listed if row["id"] == pid)["key_set"] is True
    assert secret not in out


def test_add_reads_private_key_file_and_uses_shared_connection_validation(
    monkeypatch, capsys, tmp_path: Path, client,
) -> None:
    secret_file = tmp_path / "provider.key"
    secret_file.write_text("a-secret-from-file\n", encoding="utf-8")
    secret_file.chmod(0o600)

    code, out, err = run_cli(
        monkeypatch, capsys, "assistant", "connections", "add",
        "--name", "CLI added", "--base-url", "https://api.example/v1", "--model", "model-a",
        "--key-file", str(secret_file), "--json",
    )
    assert code == 0 and not err
    result = json.loads(out)
    pid = result["connection"]["id"]
    assert pid.startswith("api-cli-added") and result["connection"]["key_set"]
    assert "a-secret-from-file" not in out
    assert get_manager()._api_key_for(pid) == "a-secret-from-file"
    api_state = client.get("/api/settings/assistant").json()
    assert any(row["id"] == pid and row["key_set"] for row in api_state["connections"])

    code, out, err = run_cli(
        monkeypatch, capsys, "assistant", "connections", "add",
        "--name", "Bad URL", "--base-url", "ftp://example/v1", "--model", "m",
        "--key-file", str(secret_file),
    )
    assert code == 1 and not out and "http://" in err and "Traceback" not in err


def test_key_file_permissions_are_checked_before_storing(monkeypatch, capsys, tmp_path: Path, client) -> None:
    key_file = tmp_path / "public.key"
    key_file.write_text("secret", encoding="utf-8")
    key_file.chmod(0o644)
    code, out, err = run_cli(
        monkeypatch, capsys, "assistant", "connections", "add",
        "--name", "Unsafe", "--base-url", "https://api.example/v1", "--model", "m",
        "--key-file", str(key_file),
    )
    assert code == 1 and not out and "chmod 600" in err
    assert not any(row["name"] == "Unsafe" for row in get_manager().list_providers())


def test_connection_update_keeps_key_and_refuses_changing_url_without_reentry(
    monkeypatch, capsys, client,
) -> None:
    pid = add_connection()
    code, out, err = run_cli(monkeypatch, capsys, "assistant", "connections", "update", pid,
                             "--model", "new-model", "--json")
    assert code == 0 and not err
    assert json.loads(out)["connection"]["model"] == "new-model"
    assert get_manager()._api_key_for(pid) == "key-not-for-output"

    code, out, err = run_cli(monkeypatch, capsys, "assistant", "connections", "update", pid,
                             "--base-url", "https://other.example/v1")
    assert code == 1 and not out and "type the API key again" in err


def test_main_connection_guards_missing_key_and_job_settings_share_model_rules(
    monkeypatch, capsys, client,
) -> None:
    mgr = get_manager()
    mgr.upsert_provider(id="api-no-key-cli", name="Missing key", kind="openai_compat",
                        model_id="m", base_url="https://api.example/v1")
    code, out, err = run_cli(monkeypatch, capsys, "assistant", "main", "api-no-key-cli")
    assert code == 1 and not out and "no saved API key" in err

    pid = add_connection()
    code, out, err = run_cli(monkeypatch, capsys, "assistant", "main", pid, "--json")
    assert code == 0 and not err and json.loads(out)["main"]["id"] == pid

    code, out, err = run_cli(monkeypatch, capsys, "assistant", "jobs", "set", "judge",
                             "--connection", "main", "--instructions", "Use concise verdict reasons.", "--json")
    assert code == 0 and not err
    row = json.loads(out)["job"]
    assert row["setting"] == "" and row["instructions"] == "Use concise verdict reasons."
    response = client.get("/api/settings/assistant")
    assert response.status_code == 200
    api_judge = next(job for job in response.json()["jobs"] if job["id"] == "judge")
    assert api_judge["instructions"] == row["instructions"] and api_judge["setting"] == ""

    code, out, err = run_cli(monkeypatch, capsys, "assistant", "jobs", "set", "teacher", "--connection", "off")
    assert code == 1 and not out and "cannot be switched off" in err
    code, out, err = run_cli(monkeypatch, capsys, "assistant", "jobs", "set", "vision", "--connection", "off")
    assert code == 0 and not err


def test_connection_delete_requires_confirmation_and_protects_main(monkeypatch, capsys, client) -> None:
    pid = add_connection()
    assistant.set_assistant_provider_id(pid)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    code, out, err = run_cli(monkeypatch, capsys, "assistant", "connections", "delete", pid)
    assert code == 1 and not out and "requires --yes" in err
    assert get_manager().get_provider(pid) is not None

    code, out, err = run_cli(monkeypatch, capsys, "assistant", "connections", "delete", pid, "--yes")
    assert code == 1 and not out and "main assistant" in err
    assert get_manager().get_provider(pid) is not None

    assistant.set_assistant_provider_id(helper.DEFAULT_HELPER_PROVIDER_ID)
    code, out, err = run_cli(monkeypatch, capsys, "assistant", "connections", "delete", pid, "--yes", "--json")
    assert code == 0 and not err and json.loads(out)["deleted"] == pid
    assert get_manager().get_provider(pid) is None


def test_assistant_help_explains_safe_key_inputs(monkeypatch, capsys) -> None:
    code, out, err = run_cli(monkeypatch, capsys, "assistant", "connections", "add", "--help")
    assert code == 0 and not err
    assert "--key-file" in out and "--key-stdin" in out and "--prompt-key" in out
    code, out, err = run_cli(monkeypatch, capsys, "assistant", "jobs", "set", "--help")
    assert code == 0 and not err and "--connection" in out and "--instructions" in out


def test_test_command_calls_the_configured_provider_without_printing_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from finetune_studio.models import providers
    from finetune_studio.webui.routes import assistant_settings

    cfg = providers.ProviderConfig(id="probe", name="Probe", kind="openai_compat", model_id="test",
                                   base_url="https://api.example/v1", api_key="private-key")
    monkeypatch.setattr(assistant_settings, "probe_connection_config", lambda _body: cfg)
    calls = {"chat": 0, "unload": 0}

    class FakeProvider:
        def __init__(self, received: providers.ProviderConfig) -> None:
            assert received.api_key == "private-key"

        def chat(self, _messages, **_kwargs) -> str:
            calls["chat"] += 1
            return "pong"

        def unload(self) -> None:
            calls["unload"] += 1

    monkeypatch.setattr(providers, "OpenAICompatProvider", FakeProvider)
    code, out, err = run_cli(monkeypatch, capsys, "assistant", "test", "probe", "--json")
    assert code == 0 and not err and json.loads(out) == {"ok": True, "reply": "pong", "latency_ms": 0}
    assert calls == {"chat": 1, "unload": 1} and "private-key" not in out


def test_models_command_authenticates_but_never_prints_the_api_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from finetune_studio.models import providers
    from finetune_studio.webui.routes import assistant_settings

    cfg = providers.ProviderConfig(id="probe", name="Probe", kind="openai_compat", model_id="test",
                                   base_url="https://api.example/v1", api_key="private-key")
    monkeypatch.setattr(assistant_settings, "probe_connection_config", lambda _body: cfg)
    observed: dict[str, str] = {}

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict[str, object]:
            return {"data": [{"id": "z-model"}, {"id": "a-model"}]}

    def fake_get(url: str, *, headers: dict[str, str], timeout: int) -> Response:
        observed.update({"url": url, "authorization": headers["Authorization"], "timeout": str(timeout)})
        return Response()

    monkeypatch.setattr("finetune_studio.cli.commands.assistant.requests.get", fake_get)
    code, out, err = run_cli(monkeypatch, capsys, "assistant", "models", "probe", "--json")
    assert code == 0 and not err and json.loads(out) == {"models": ["a-model", "z-model"]}
    assert observed == {"url": "https://api.example/v1/models", "authorization": "Bearer private-key", "timeout": "20"}
    assert "private-key" not in out
