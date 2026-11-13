"""A new version keeps what its package is for (walkthrough 2026-11-13, B1/M1/M2/M4/M11).

B1: a public bot's next version became the internal package (all documents, originals on) without a word, and
update.sh installed it into the running public bot. M1: a new version got a new generated passphrase and the
running copy no longer started after update.sh. M2: update.sh kept every old rag.config.json value, dropping the new
version's settings. M4: after a build the form silently switched to "new version of" it. M11: the README's MCP
commands had no --keyfile for an encrypted package.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from finetune_studio.smart_brain import package_meta as pm
from finetune_studio.smart_brain import package_versions as pv
from finetune_studio.smart_brain import scope as sc
from finetune_studio.smart_brain import standalone_server as ss
from tests.test_brain_server_package import corpus_dir  # noqa: F401  (fixture)
from tests.test_package_scope import DOCS, PASS, export_env  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]


def test_widening_names_every_way_a_version_opens_up() -> None:
    base = {"version": "1.0.0", "scope_spec": {"include": ["public/"]}, "include_originals": False,
            "hide_file_names": True, "audience": "public", "encrypted": True}
    same = pv.widening(base, DOCS, {"c1", "c2"}, include_originals=False, hide_file_names=True, audience="public",
                       encrypt=True)
    assert same == []
    wide = pv.widening(base, DOCS, None, include_originals=True, hide_file_names=False, audience="internal",
                       encrypt=False)
    assert [w["code"] for w in wide] == ["scope", "originals", "file_names", "audience", "encryption"]
    assert "1 document(s) outside version 1.0.0's scope (internal/)" in wide[0]["text"]
    # A version narrowed or kept inside the scope is never "wider"; a new file inside public/ is not either.
    assert pv.widening(base, DOCS, {"c1"}, include_originals=False, hide_file_names=True, audience="public",
                       encrypt=True) == []


def test_an_older_base_without_a_recorded_scope_uses_its_documents_folders() -> None:
    legacy = {"version": "1.0.0", "scope": "paths public/", "documents": [{"path": "public/ledger.txt"}],
              "originals": 0, "hide_file_names": True}
    scope = pv.base_scope(legacy)
    assert scope is not None and scope.include == ["public/"]
    assert {d.document_id for d in DOCS if scope.selects(d)} == {"c1", "c2"}
    assert pv.base_scope({"scope": "all documents", "documents": [{"path": "a/b.txt"}]}) is None
    st = pv.base_settings(legacy)
    assert st["include_originals"] is False and st["hide_file_names"] is True and st["encrypted"] is True
    assert pv.inherit_scope(None, legacy).include == ["public/"]
    asked = sc.DocumentScope(include=["internal/"])
    assert pv.inherit_scope(asked, legacy) is asked


def test_new_version_inherits_and_refuses_to_widen_without_confirmation(client, export_env) -> None:  # noqa: F811
    pid = export_env
    url = f"/api/projects/{pid}/brain/standalone"
    first = client.post(url, json={"semantic_search": False, "passphrase": PASS, "scope": {"include": ["public/"]},
                                   "audience": "public", "hide_file_names": True, "include_originals": False,
                                   "name": "Highmere public bot"})
    assert first.status_code == 200, first.text
    pkg = first.json()["package"]["package_id"]
    settings = client.get(f"/api/projects/{pid}/brain/packages").json()["packages"][0]["settings"]
    assert settings == {"scope": {"include": ["public/"], "exclude": [], "tags": [], "document_ids": []},
                        "include_originals": False, "hide_file_names": True, "audience": "public", "encrypted": True}

    # The walkthrough's request: the form's defaults (all documents, originals on, names shown, internal).
    wide = {"semantic_search": False, "passphrase": PASS, "new_version_of": pkg, "scope": {},
            "include_originals": True, "hide_file_names": False, "audience": "internal"}
    refused = client.post(url, json=wide)
    assert refused.status_code == 409, refused.text
    body = refused.json()
    assert body["code"] == "widens_package"
    assert [w["code"] for w in body["widens"]] == ["scope", "originals", "file_names", "audience"]
    assert "internal/" in body["detail"]
    assert pm.latest(pm.package_dir(pid), pkg)["version"] == "1.0.0"         # nothing was built

    # Leaving the settings open inherits the base's: same scope, no originals, names hidden, public, same name.
    kept = client.post(url, json={"semantic_search": False, "passphrase": PASS, "new_version_of": pkg})
    assert kept.status_code == 200, kept.text
    j = kept.json()
    assert j["package"]["version"] == "1.1.0" and j["scope"]["documents"] == 2 and j["original_files"] == 0
    assert j["widens"] == [] and j["filename"].startswith("agent-test-scope-highmere-public-bot-v1.1.0-")
    row = pm.latest(pm.package_dir(pid), pkg)
    assert row["hide_file_names"] is True and row["audience"] == "public" and row["scope_spec"]["include"] == ["public/"]

    # M1: an encrypted new version needs the current passphrase (or a new one on purpose).
    blank = client.post(url, json={"semantic_search": False, "new_version_of": pkg})
    assert blank.status_code == 422 and blank.json()["code"] == "passphrase_required"
    wrong = client.post(url, json={"semantic_search": False, "new_version_of": pkg, "passphrase": "another one!!"})
    assert wrong.status_code == 422 and wrong.json()["code"] == "passphrase_mismatch"

    # Confirmed: built, recorded in CHANGES.md and in package.json for update.sh.
    allowed = client.post(url, json={**wide, "allow_widen": True})
    assert allowed.status_code == 200, allowed.text
    out = pm.package_dir(pid) / allowed.json()["filename"]
    with tarfile.open(out) as tf:
        root = next(n for n in tf.getnames() if n.endswith("/package.json"))
        public = json.loads(tf.extractfile(root).read())
        changes = tf.extractfile(root.replace("package.json", "CHANGES.md")).read().decode()
    assert public["widens"] == ["scope", "originals", "file_names", "audience"]
    assert public["policy"]["scope"] == "all" and public["policy"]["audience"] == "internal"
    assert "Wider than version 1.1.0" in changes and "internal/" not in changes      # names stay encrypted
    assert "emberfall" not in json.dumps(public)
    fresh = client.post(url, json={"semantic_search": False, "new_version_of": pkg, "new_passphrase": True})
    assert fresh.status_code == 200 and fresh.json()["passphrase"], fresh.text
    with tarfile.open(pm.package_dir(pid) / fresh.json()["filename"]) as tf:
        root = next(n for n in tf.getnames() if n.endswith("/package.json"))   # each version has its dated folder
        assert json.loads(tf.extractfile(root).read())["passphrase_changed"] is True


def test_export_form_loads_base_settings_and_resets_after_a_build(client, export_env) -> None:  # noqa: F811
    page = client.get(f"/projects/{export_env}/export").text
    for marker in ("function applyBaseSettings(st)", 'id="mcp-newpass"', "new_passphrase:",
                   "e.code === 'widens_package'", "body.allow_widen = true",
                   "document.getElementById('mcp-base').value = '';   // the next export is a new package"):
        assert marker in page, marker
    assert "loadPackageHistory(d.package.package_id)" not in page                    # M4: no silent switch


def test_merge_config_keeps_only_what_the_user_changed() -> None:
    shipped = {"port": 8899, "top_k": 10, "answer": False, "llm_model": "", "hide_file_names": False}
    local = {**shipped, "port": 9555}                                       # the user moved the port only
    new = {"port": 8899, "top_k": 8, "answer": True, "llm_model": "qwen3:8b", "hide_file_names": True}
    merged, kept, taken = ss.merge_config(local, shipped, new)
    assert merged == {**new, "port": 9555} and kept == ["port"]
    assert taken == ["answer", "hide_file_names", "llm_model", "top_k"]
    # A package from before the shipped copy: only where-and-how-it-runs keys survive.
    merged, kept, _ = ss.merge_config({**shipped, "port": 9555, "answer": False}, None, new)
    assert merged["port"] == 9555 and merged["answer"] is True and kept == ["port", "top_k"]


def _fake_venv(root: Path) -> None:
    (root / ".venv" / "bin").mkdir(parents=True)
    py = root / ".venv" / "bin" / "python"
    py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    pip = root / ".venv" / "bin" / "pip"
    pip.write_text("#!/bin/sh\nexit 0\n")
    pip.chmod(0o755)


def _run(root: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_")}
    return subprocess.run(["bash", "update.sh", *args], check=False, cwd=root, env=env, capture_output=True, text=True,
                          timeout=180)


def test_update_refuses_a_wider_version_and_needs_the_passphrase(client, export_env, tmp_path: Path) -> None:  # noqa: F811
    pid = export_env
    url = f"/api/projects/{pid}/brain/standalone"
    v1 = client.post(url, json={"semantic_search": False, "passphrase": PASS, "scope": {"include": ["public/"]},
                                "audience": "public", "hide_file_names": True}).json()
    pkg = v1["package"]["package_id"]
    with tarfile.open(pm.package_dir(pid) / v1["filename"]) as tf:
        tf.extractall(tmp_path / "srv", filter="data")
    root = next((tmp_path / "srv").rglob("server.py")).parent
    _fake_venv(root)
    assert (root / "rag.config.shipped.json").is_file()
    readme = (root / "README.md").read_text()
    assert (f"claude mcp add {json.loads((root / 'package.json').read_text())['slug']} -- bash /ABSOLUTE/PATH/run-mcp.sh "
            "--keyfile /ABSOLUTE/PATH/passphrase.txt") in readme
    assert '"/ABSOLUTE/PATH/run-mcp.sh", "--keyfile", "/ABSOLUTE/PATH/passphrase.txt"' in readme
    assert "--allow-widen" in readme
    cfg = json.loads((root / "rag.config.json").read_text())
    (root / "rag.config.json").write_text(json.dumps({**cfg, "port": 9555}))
    wide = client.post(url, json={"semantic_search": False, "passphrase": PASS, "new_version_of": pkg, "scope": {},
                                  "include_originals": True, "audience": "internal", "allow_widen": True,
                                  "answer_enabled": True, "llm_base_url": "http://127.0.0.1:11434/v1",
                                  "llm_model": "qwen3:8b"}).json()
    archive = str((pm.package_dir(pid) / wide["filename"]).resolve())

    keyfile = tmp_path / "key.txt"
    keyfile.write_text(PASS)
    refused = _run(root, archive, "--keyfile", str(keyfile))
    assert refused.returncode == 2 and "opens up more than the installed 1.0.0" in refused.stderr
    assert "original files: none -> on" in refused.stderr and "--allow-widen" in refused.stderr
    no_key = _run(root, archive, "--allow-widen")
    assert no_key.returncode == 2 and "give --keyfile" in no_key.stderr, no_key.stderr
    assert json.loads((root / "package.json").read_text())["version"] == "1.0.0"
    (root / "passphrase.txt").write_text(PASS)                    # the default keyfile, as setup.sh writes it
    done = _run(root, archive, "--allow-widen")
    assert done.returncode == 0, done.stderr
    assert "kept your settings: port" in done.stderr and "WIDER" in done.stderr
    merged = json.loads((root / "rag.config.json").read_text())
    assert merged["port"] == 9555 and merged["answer"] is True and merged["llm_model"] == "qwen3:8b"   # M2
    assert json.loads((root / "package.json").read_text())["version"] == "1.1.0"


def test_cli_has_the_confirmation_flags() -> None:
    out = subprocess.run([sys.executable, "-m", "finetune_studio.cli", "brain", "export", "--help"],
                         check=False, capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    for flag in ("--allow-widen", "--new-passphrase", "--audience"):
        assert flag in out.stdout


def test_setup_sh_writes_the_keyfile_into_the_mcp_entry() -> None:
    from finetune_studio.smart_brain.server_package import _SETUP_SH
    assert 'MCP_ARGS="\\"$PKG_DIR/run-mcp.sh\\", \\"--keyfile\\", \\"${KEYFILE:-$PKG_DIR/passphrase.txt}\\""' in _SETUP_SH
    assert '( umask 077; printf' in _SETUP_SH and '"$PKG_DIR/passphrase.txt" )' in _SETUP_SH
