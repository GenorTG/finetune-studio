"""Connection recipes must launch the exported package, not pretend it is an LLM."""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/static/js/connect.js"
CLIENTS = ("claude-code", "claude-desktop", "cursor", "vscode", "windsurf", "gemini",
           "lmstudio", "openwebui", "chatgpt", "ollama")


def recipe(client="cursor", folder="/ABSOLUTE/PATH/TO/package",
           keyfile="/ABSOLUTE/PATH/TO/passphrase.txt", address="http://127.0.0.1:8899"):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute the browser recipe builder")
    result = subprocess.run(
        [node, "-e", ("const {connectRecipes} = require(process.argv[1]);"
         "try { console.log(JSON.stringify(connectRecipes(...JSON.parse(process.argv[2])))); }"
         "catch(e) { console.log(JSON.stringify({error:e.message})); }"),
         str(SCRIPT), json.dumps([client, folder, keyfile, address])],
        capture_output=True, text=True, check=True, timeout=10,
    )
    return json.loads(result.stdout)


def test_connect_is_reachable_and_package_navigation_preserves_exports(client):
    from finetune_studio import db

    response = client.get("/connect")
    assert response.status_code == 200
    assert "Connect your Smart Brain" in response.text
    assert 'src="/static/js/connect.js?v=1"' in response.text
    assert "Client connection not checked" in response.text
    assert "/v1/chat/completions" in response.text
    assert "not shipped" in response.text
    project = db.create_project(name="connect navigation")
    package = client.get(f"/projects/{project['id']}/export")
    assert package.status_code == 200
    assert 'id="package-connect" href="/connect"' in package.text
    assert "Heretic" in package.text
    assert 'data-tab="connect"' in client.get("/projects").text


@pytest.mark.parametrize("client_name", CLIENTS)
def test_every_client_has_an_explicit_setup_or_limitation(client_name):
    result = recipe(client_name)
    assert result["instructions"]
    assert result["docs"].startswith("https://")
    assert result["config"]
    assert "/health" in result["test"] and "/mcp" in result["test"]
    assert "/v1/chat/completions" not in result["config"]
    assert result["availableSetup"] is (client_name not in ("chatgpt", "ollama"))


def test_standard_stdio_matches_exported_package_contract():
    from finetune_studio.smart_brain.server_package import _mcp_config_example

    expected = json.loads(_mcp_config_example("smart-brain", "package", encrypted=True))
    assert json.loads(recipe()["config"]) == expected


def test_vscode_uses_servers_not_mcpservers():
    result = json.loads(recipe("vscode")["config"])
    assert result["servers"]["smart-brain"] == {
        "type": "stdio", "command": "bash",
        "args": ["/ABSOLUTE/PATH/TO/package/run-mcp.sh", "--keyfile", "/ABSOLUTE/PATH/TO/passphrase.txt"],
    }


def test_shell_quoting_preserves_paths_and_no_shell_expansion():
    folder = "/home/a user's/brain $(touch bad) `id`"
    keyfile = "/keys/it's a passphrase.txt"
    result = recipe("claude-code", folder, keyfile)
    assert shlex.split(result["config"]) == [
        "claude", "mcp", "add", "smart-brain", "--", "bash",
        folder + "/run-mcp.sh", "--keyfile", keyfile,
    ]
    assert shlex.split(result["start"]) == ["bash", folder + "/run-http.sh", "--keyfile", keyfile]


def test_http_checks_keep_proxy_prefix_and_initialize_contract():
    result = recipe("openwebui", address="https://brain.example/team/mcp/")
    assert result["config"] == "https://brain.example/team/mcp"
    health, initialize = result["test"].split("\n\n")
    assert shlex.split(health)[-1] == "https://brain.example/team/health"
    command = shlex.split(initialize)
    assert command[-1] == "https://brain.example/team/mcp"
    payload = json.loads(command[command.index("--data") + 1])
    assert payload["method"] == "initialize"
    assert payload["params"]["protocolVersion"] == "2024-11-05"
    assert payload["params"]["clientInfo"]["name"] == "connect-check"
    assert "${RAG_AUTH_TOKEN:-}" in result["test"]
    assert "--max-time 15" in result["test"]


@pytest.mark.parametrize("overrides", [
    {"folder": "relative/package"}, {"keyfile": ""}, {"folder": "/tmp/bad\ncommand"},
    {"address": "file:///etc/passwd"}, {"address": "https://user:secret@example.com"},
    {"address": "https://example.com/?token=secret"}, {"address": "https://example.com/#token"},
])
def test_invalid_paths_and_credential_urls_are_rejected(overrides):
    assert "error" in recipe(**overrides)


def test_guide_can_navigate_highlight_and_explain_connect():
    from finetune_studio.guide import registry
    from finetune_studio.guide.kb import load_kb
    from finetune_studio.guide.settings_help import explain_setting

    assert registry.resolve_page_url("connect", None) == ("/connect", None)
    assert registry.CONTROLS["connect.copy"].selector == "#connect-copy"
    assert load_kb()["connect"].page == "connect"
    for name in ("connect_client", "connect_path", "connect_keyfile", "connect_url"):
        assert ("connect", name) in registry.FIELDS
        assert explain_setting(name)
