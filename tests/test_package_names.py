"""Unique, dated package names and the package's hosted metadata (walk m5).

Archive and folder: ``<project>-<package>-v<version>-<YYYYMMDD>``; the install name (systemd unit, MCP server name)
is ``<package>-<id8>``, the same for every version of one package and never the same for two packages. The package
says who it is everywhere: ``package.json``, ``GET /version`` (public), ``/health``, the ``X-Package-Version``
header, MCP ``serverInfo`` and instructions, ``rag_info``, the chat page and the README. An install from before the
stable name (every package extracted to ``rag-pack/``, unit ``fts-brain-rag-pack``) updates in place and moves to
the stable name only on ``--rename``."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from finetune_studio.smart_brain import package_meta as pm
from finetune_studio.smart_brain.server_package import build_package
from tests.test_brain_server_package import corpus_dir  # noqa: F401  (fixture)
from tests.test_package_versions import DOCS, FAST, PASS, _fake_venv

CREATED = "2026-11-13T10:00:00+0100"


def _record(name: str, project: str = "Quillfen", base: dict | None = None, docs=None) -> dict:
    rec = pm.make_record(base=base, documents=docs or DOCS)
    rec["created"] = CREATED
    return pm.name_record(rec, base, name=name, project=project)


def _build(corpus: Path, out_dir: Path, record: dict, **kw):
    diff = pm.changes(None, record["documents"])
    record = {**record, "changes": pm.summary(diff)}
    out = out_dir / pm.package_filename(record, ext="tar.gz")
    return build_package(corpus, out, name=record["name"], config=kw.pop("config", FAST), passphrase=PASS,
                         package=record, changes_md=pm.changes_markdown(record, diff),
                         changes_md_public=pm.changes_markdown(record, diff, names=False), **kw)


def _extract(archive: Path, dest: Path) -> Path:
    with tarfile.open(archive) as tf:
        tf.extractall(dest, filter="data")
    return next(dest.rglob("server.py")).parent


def test_archive_folder_and_install_names(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    staff, public = _record("Staff handbook"), _record("Public bot")
    a = _build(corpus_dir, tmp_path, staff)
    b = _build(corpus_dir, tmp_path, public)
    assert a.path.name == "quillfen-staff-handbook-v1.0.0-20261113.tar.gz"
    assert a.root == "quillfen-staff-handbook-v1.0.0-20261113" != b.root
    root = _extract(a.path, tmp_path / "x")
    assert root.name == a.root
    meta = json.loads((root / "package.json").read_text())
    assert meta["name"] == "Staff handbook" and meta["slug"] == "staff-handbook-" + staff["package_id"][:8]
    assert meta["project"] == "Quillfen" and meta["built"] == "2026-11-13" and meta["documents"] == 3
    assert meta["layers"]["keyword"] is True and meta["manifest_sha256"] == pm.manifest_hash(DOCS)
    assert meta["studio_version"]
    assert "ledger" not in (root / "package.json").read_text()           # no document names beside an encrypted corpus
    readme = (root / "README.md").read_text()
    assert readme.splitlines()[2].startswith("**Staff handbook · version 1.0.0 · built 2026-11-13**")
    assert f"fts-brain-{meta['slug']}" in readme and "ledger" not in readme
    mcp = json.loads((root / "mcp-config.example.json").read_text())
    assert list(mcp["mcpServers"]) == [meta["slug"]]
    setup = (root / "setup.sh").read_text()
    assert '"slug"' in setup and "--rename" in setup and "WorkingDirectory=$PKG_DIR" in setup


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str, token: str = ""):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.headers, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, e.headers, json.loads(e.read() or b"{}")


def test_running_package_reports_its_version(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    rec = _record("Staff handbook")
    root = _extract(_build(corpus_dir, tmp_path, rec).path, tmp_path / "x")
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_")} | {
        "RAG_PASSPHRASE": PASS, "RAG_AUTH_TOKEN": "tok-123456789"}
    log = open(tmp_path / "srv.log", "w")  # noqa: SIM115
    proc = subprocess.Popen([sys.executable, "-I", "server.py", "--http", str(port)], cwd=root, env=env,
                            stdout=subprocess.DEVNULL, stderr=log)
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                code, headers, body = _get(base + "/version")
                break
            except OSError:
                time.sleep(0.25)
        # /version needs no token and names no document; /health needs the token and carries the same block.
        assert code == 200 and body["version"] == "1.0.0" and body["name"] == "Staff handbook"
        assert body["slug"] == rec["slug"] and body["built"] == "2026-11-13" and body["encrypted"] is True
        assert headers["X-Package-Version"] == "1.0.0" and "documents_list" not in body
        assert "ledger" not in json.dumps(body)
        assert _get(base + "/health")[0] == 401
        code, headers, health = _get(base + "/health", "tok-123456789")
        assert code == 200 and health["version"]["version"] == "1.0.0" and headers["X-Package-Version"] == "1.0.0"
        req = urllib.request.Request(base + "/", headers={})
        with urllib.request.urlopen(req, timeout=10) as r:
            page = r.read().decode()
        assert 'id="pkg-version"' in page
        with urllib.request.urlopen(base + "/app.js", timeout=10) as r:
            assert "fetch('/version')" in r.read().decode()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        log.close()


def test_mcp_names_the_package(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    rec = _record("Staff handbook")
    root = _extract(_build(corpus_dir, tmp_path, rec).path, tmp_path / "x")
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_")} | {"RAG_PASSPHRASE": PASS}
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}},
            {"jsonrpc": "2.0", "id": 2, "method": "server/discover",
             "params": {"_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "rag_info", "arguments": {}}}]
    done = subprocess.run([sys.executable, "-I", "server.py", "--mcp"], cwd=root, env=env, timeout=120,
                          input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, text=True)
    replies = {r["id"]: r for r in map(json.loads, filter(None, done.stdout.splitlines()))}
    init = replies[1]["result"]
    assert init["serverInfo"] == {"name": rec["slug"], "title": "Staff handbook", "version": "1.0.0"}
    assert "Staff handbook · 1.0.0 · 2026-11-13" in init["instructions"]
    disc = replies[2]["result"]
    assert "1.0.0" in disc["instructions"] and disc["_meta"]["io.modelcontextprotocol/serverInfo"]["version"] == "1.0.0"
    info = json.loads(replies[3]["result"]["content"][0]["text"])
    assert info["version"]["version"] == "1.0.0" and info["version"]["slug"] == rec["slug"]


def _fake_systemctl(tmp_path: Path) -> tuple[Path, Path]:
    """A ``systemctl`` that logs its calls and says every unit is active (setup.sh/update.sh only call it)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "systemctl.log"
    stub = bin_dir / "systemctl"
    stub.write_text(f'#!/bin/sh\necho "$@" >> "{calls}"\nexit 0\n')
    stub.chmod(0o755)
    return bin_dir, calls


def test_old_rag_pack_install_updates_in_place_then_renames(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    # An install from before the stable name: the folder is rag-pack/, the service fts-brain-rag-pack, the MCP
    # entry "rag-pack" (that is what 0.5 packages did); its package.json has no slug.
    v1 = _record("Staff handbook")
    old = _extract(_build(corpus_dir, tmp_path, v1).path, tmp_path / "x")
    root = tmp_path / "srv" / "rag-pack"
    root.parent.mkdir()
    old.rename(root)
    meta = json.loads((root / "package.json").read_text())
    for key in ("slug", "name", "root"):
        meta.pop(key)
    (root / "package.json").write_text(json.dumps(meta, indent=2))
    _fake_venv(root)
    (root / "passphrase.txt").write_text(PASS)
    home = tmp_path / "home"
    units = home / ".config" / "systemd" / "user"
    units.mkdir(parents=True)
    (units / "fts-brain-rag-pack.service").write_text(
        f"[Unit]\nDescription=Finetune Studio RAG pack (rag-pack)\n[Service]\nWorkingDirectory={root}\n"
        f"ExecStart={root}/.venv/bin/python {root}/server.py --http --port 8911\n")
    (root / "MCP-ENTRY.txt").write_text('{\n  "mcpServers": {\n    "rag-pack": {\n      "command": "bash"\n    }\n  }\n}\n')
    bin_dir, calls = _fake_systemctl(tmp_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_")} | {
        "HOME": str(home), "PATH": f"{bin_dir}:{os.environ['PATH']}", "XDG_RUNTIME_DIR": str(tmp_path)}

    v2 = _record("Staff handbook", base=v1)
    (tmp_path / "out2").mkdir()
    archive = _build(corpus_dir, tmp_path / "out2", v2).path
    up = subprocess.run(["bash", "update.sh", str(archive)], cwd=root, env=env, capture_output=True, text=True,
                        timeout=180)
    assert up.returncode == 0, up.stdout + up.stderr
    assert json.loads((root / "package.json").read_text())["version"] == "1.1.0"
    assert f"installs as '{v1['slug']}'" in up.stdout and "bash update.sh --rename" in up.stdout
    assert "restart fts-brain-rag-pack.service" in calls.read_text()        # the running copy keeps working as is
    assert (units / "fts-brain-rag-pack.service").is_file()                  # nothing renamed without asking

    ren = subprocess.run(["bash", "update.sh", "--rename"], cwd=root, env=env, capture_output=True, text=True,
                         timeout=60)
    assert ren.returncode == 0, ren.stdout + ren.stderr
    new_unit = units / f"fts-brain-{v1['slug']}.service"
    assert new_unit.is_file() and not (units / "fts-brain-rag-pack.service").exists()
    assert f"WorkingDirectory={root}" in new_unit.read_text() and v1["slug"] in new_unit.read_text()
    assert f'"{v1["slug"]}": {{' in (root / "MCP-ENTRY.txt").read_text()
    log = calls.read_text()
    assert "disable --now fts-brain-rag-pack.service" in log and f"start fts-brain-{v1['slug']}.service" in log
    again = subprocess.run(["bash", "setup.sh", "--rename"], cwd=root, env=env, capture_output=True, text=True,
                           timeout=60)
    assert "Nothing to rename" in again.stdout


def test_two_packages_on_one_host_get_distinct_units(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    first, second = _record("Staff handbook"), _record("Staff handbook")      # same name, two packages
    assert first["slug"] != second["slug"]
    home = tmp_path / "home"
    (home / ".config" / "systemd" / "user").mkdir(parents=True)
    bin_dir, calls = _fake_systemctl(tmp_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_")} | {
        "HOME": str(home), "PATH": f"{bin_dir}:{os.environ['PATH']}", "XDG_RUNTIME_DIR": str(tmp_path)}
    (tmp_path / "pw").write_text(PASS)
    names = []
    for i, rec in enumerate((first, second)):
        out = tmp_path / f"o{i}"
        out.mkdir()
        root = _extract(_build(corpus_dir, out, rec).path, tmp_path / f"x{i}")
        _fake_venv(root)
        done = subprocess.run(["bash", "setup.sh", "--yes", "--service", "--port", str(8910 + i),
                               "--keyfile", str(tmp_path / "pw")], cwd=root, env=env, capture_output=True,
                              text=True, timeout=120, input="")
        assert done.returncode == 0, done.stdout + done.stderr
        names.append(json.loads((root / "MCP-ENTRY.txt").read_text())["mcpServers"])
    assert list(names[0]) == [first["slug"]] and list(names[1]) == [second["slug"]]
    units = sorted(p.name for p in (home / ".config" / "systemd" / "user").iterdir())
    assert units == sorted(f"fts-brain-{r['slug']}.service" for r in (first, second))
