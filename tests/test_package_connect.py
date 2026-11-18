"""Exercise real exported Connect assets and the MCPB stdio bridge, without client certification claims."""
from __future__ import annotations

import io
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from finetune_studio.smart_brain.connect_bundle import build_mcpb
from finetune_studio.smart_brain.server_package import build_package
from tests import test_brain_server_package as package_tests


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    return package_tests.corpus_dir.__wrapped__(tmp_path)


@pytest.fixture()
def exported_package(corpus_dir: Path, tmp_path: Path) -> Path:
    archive = build_package(corpus_dir, tmp_path / "package.tar.gz", encrypt=False, name="Private Test Name")
    return package_tests._extract(archive.path, tmp_path / "unpacked")


@pytest.fixture()
def hosted_package(exported_package: Path) -> Iterator[str]:
    """Run the exported file, not the studio/imported handler."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (exported_package / "http.log").open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-I", str(exported_package / "server.py"), "--http", str(port), "--host", "127.0.0.1"],
            env={**os.environ, "RAG_AUTH_TOKEN": "connect-check-token"},
            stdout=subprocess.DEVNULL, stderr=log,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(60):
                try:
                    with urllib.request.urlopen(base + "/connect", timeout=2):
                        break
                except urllib.error.HTTPError:
                    raise
                except OSError:
                    if proc.poll() is not None:
                        pytest.fail((exported_package / "http.log").read_text())
                    time.sleep(0.1)
            else:
                pytest.fail("exported HTTP server did not become ready")
            yield base
        finally:
            proc.terminate()
            proc.wait(timeout=10)


def test_public_assets_and_authenticated_mcp(hosted_package: str) -> None:
    for route, needle in (("/connect", b"Test package MCP"), ("/connect.js", b"connectRecipes"),
                          ("/connect-page.js", b"notifications/initialized")):
        with urllib.request.urlopen(hosted_package + route, timeout=5) as response:
            assert needle in response.read()
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
    with urllib.request.urlopen(hosted_package + "/", timeout=5) as response:
        assert b'href="connect"' in response.read()
    with urllib.request.urlopen(hosted_package + "/smart-brain.mcpb", timeout=5) as response:
        assert "attachment" in response.headers["Content-Disposition"]
        assert zipfile.is_zipfile(io.BytesIO(response.read()))
    with pytest.raises(urllib.error.HTTPError) as unauthorized:
        urllib.request.urlopen(hosted_package + "/health", timeout=5)
    assert unauthorized.value.code == 401
    request = urllib.request.Request(hosted_package + "/mcp", data=json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "rag_info", "arguments": {}}
    }).encode(), headers={"Authorization": "Bearer connect-check-token", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        result = json.loads(response.read())["result"]
        assert result["isError"] is False and result["content"]
    request.add_header("Origin", "https://foreign.example")
    with pytest.raises(urllib.error.HTTPError) as foreign:
        urllib.request.urlopen(request, timeout=5)
    assert foreign.value.code == 403


@pytest.mark.parametrize("encrypted", [False, True])
def test_bridge_from_export_runs_real_mcp(corpus_dir: Path, tmp_path: Path, encrypted: bool) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the MCPB bridge")
    archive = build_package(corpus_dir, tmp_path / "package.tar.gz", encrypt=encrypted,
                            passphrase="connect-test-secret" if encrypted else None, name="Secret Corpus Name")
    root = package_tests._extract(archive.path, tmp_path / "unpacked")
    # Reuse the test interpreter's installed dependencies, just as run-mcp.sh expects after install.sh.
    (root / ".venv").symlink_to(sys.prefix, target_is_directory=True)
    keyfile = tmp_path / "key ' quoted.txt"
    keyfile.write_text("connect-test-secret\n")
    # Extraction location must not change the bridge's CommonJS interpretation.
    (tmp_path / "package.json").write_text('{"type": "module"}\n')
    extension = tmp_path / "extension"
    with zipfile.ZipFile(root / "smart-brain.mcpb") as bundle:
        assert set(bundle.namelist()) == {"manifest.json", "bridge.cjs", "README.md"}
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["compatibility"]["platforms"] == ["linux"]
        assert manifest["server"]["type"] == "node"
        assert "Secret Corpus Name" not in bundle.read("manifest.json").decode()
        assert "connect-test-secret" not in b"".join(bundle.read(n) for n in bundle.namelist()).decode()
        bundle.extractall(extension)
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "rag_info", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "rag_search", "arguments": {"query": "Highmere Stair"}}},
    ]
    # Resolve exactly the variables collected by an MCPB host; execute without shell interpolation.
    substitutions = {"${__dirname}": str(extension), "${user_config.package_dir}": str(root),
                     "${user_config.keyfile}": str(keyfile)}
    args = [substitutions.get(arg, arg.replace("${__dirname}", str(extension)))
            for arg in manifest["server"]["mcp_config"]["args"]]
    run = subprocess.run([node, *args], input="".join(json.dumps(req) + "\n" for req in requests),
                         capture_output=True, text=True, timeout=45, check=False)
    assert run.returncode == 0, run.stderr
    replies = {row["id"]: row for row in map(json.loads, run.stdout.splitlines())}
    assert replies[1]["result"]["serverInfo"]
    assert {"rag_info", "rag_search"} <= {tool["name"] for tool in replies[2]["result"]["tools"]}
    assert replies[3]["result"]["isError"] is False
    assert "nine crowns" in replies[4]["result"]["content"][0]["text"]


def test_bridge_prerequisite_failure_has_no_protocol_stdout(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the MCPB bridge")
    with zipfile.ZipFile(io.BytesIO(build_mcpb())) as bundle:
        bundle.extractall(tmp_path)
    run = subprocess.run([node, str(tmp_path / "bridge.cjs"), str(tmp_path / "missing"), str(tmp_path / "key")],
                         capture_output=True, text=True, timeout=5, check=False)
    assert run.returncode == 1 and not run.stdout
    assert "run bash install.sh" in run.stderr


def test_missing_assets_are_an_honest_404(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from finetune_studio.smart_brain import standalone_server as srv

    monkeypatch.setattr(srv, "__file__", str(tmp_path / "server.py"))
    server = srv._Server(("127.0.0.1", 0), srv.make_handler(None))
    import threading

    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as response:
            urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/connect", timeout=5)
        assert response.value.code == 404
        assert "export a new package" in response.value.read().decode()
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("failure", ["", "unauthorized", "missing-tools", "rpc-error", "insecure"])
def test_browser_check_never_probes_the_recipe_address_or_stores_tokens(failure: str) -> None:
    """Execute the shipped JS: editable HTTP recipes cannot redirect bearer-token checks."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the package browser check")
    script = Path(__file__).resolve().parents[1] / "src/finetune_studio/smart_brain/package_connect.js"
    harness = r"""
const vm = require('node:vm'), fs = require('node:fs');
const failure = process.argv[2], nodes = {}, calls = [];
for (const id of ['connect-url', 'package-token', 'package-check', 'package-check-status']) {
  nodes[id] = {value: '', textContent: '', disabled: false, addEventListener: (_, fn) => {nodes[id].click = fn;}};
}
nodes['package-token'].value = 'synthetic-token';
const href = (failure === 'insecure' ? 'http:' : 'https:') + '//package.example/tenant/connect';
const location = new URL(href);
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
  document: {getElementById: id => nodes[id]}, location, URL, AbortSignal,
  // No localStorage/sessionStorage provided: any attempted persistence breaks the check.
  fetch: async (url, options) => {
    calls.push({url: url.href, auth: options.headers.Authorization === 'Bearer synthetic-token'});
    const body = options.body ? JSON.parse(options.body) : {};
    let data = {ok: true};
    if (body.method === 'initialize') data = {result: {serverInfo: {name: 'test'}}};
    if (body.method === 'tools/list') data = {result: {tools: failure === 'missing-tools' ? [] : [{name:'rag_info'},{name:'rag_search'}]}};
    if (body.method === 'tools/call') data = failure === 'rpc-error' ? {error: {message: 'failure'}} : {result: {isError:false,content:[{type:'text',text:'{}'}]}};
    return {ok: failure !== 'unauthorized', status: failure === 'unauthorized' ? 401 : body.method === 'notifications/initialized' ? 202 : 200, json: async () => data};
  }
});
nodes['connect-url'].value = 'https://attacker.example/private';
nodes['package-check'].click().then(() => console.log(JSON.stringify({
  status: nodes['package-check-status'].textContent, disabled: nodes['package-check'].disabled, calls
})));
"""
    run = subprocess.run([node, "-e", harness, str(script), failure], capture_output=True, text=True,
                         timeout=10, check=True)
    result = json.loads(run.stdout)
    assert not result["disabled"]
    assert ("check passed" in result["status"]) is (not failure)
    assert all(call["url"] in ("https://package.example/tenant/health", "https://package.example/tenant/mcp")
               and call["auth"] for call in result["calls"])
    if failure == "insecure":
        assert not result["calls"] and "HTTPS" in result["status"]
    if failure == "unauthorized":
        assert "401" in result["status"]
