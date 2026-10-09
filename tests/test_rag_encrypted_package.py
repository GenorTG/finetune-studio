"""Encrypted-at-rest + configurable Portable RAG export package.

Covers the security/config contract of ``build_package`` and the shipped
``server.py``: encrypt->serve->query parity with the plaintext baseline, no
plaintext anywhere in the archive, fail-closed on a wrong passphrase / flipped
byte, flag > env > file > default precedence, the loopback-bind rule, and no
plaintext left in the system temp dir after a build or a server start.

The shipped server is executed as a real subprocess (same interpreter).
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import urllib.error
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from finetune_studio.data.rag_portable import export_config as ec
from finetune_studio.data.rag_portable.export_config import RagExportConfig
from finetune_studio.data.rag_portable.mcp_package import PackageResult, build_package
from finetune_studio.data.rag_portable.rag_container import (
    ContainerError,
    ContainerReader,
    ContainerWriter,
    WrongPassphraseOrTampered,
)
from tests.test_rag_mcp_package import _IDS, _TEXTS, corpus_dir  # noqa: F401  (fixture)

PASS = "correct horse battery staple"
FAST = RagExportConfig(kdf_log_n=10)  # cheap scrypt for tests
QUERIES = ["oath of salt ledger", "highmere crowns", "emberfall meteoric", "zzz-no-hit"]
NEEDLES = [
    "oath of salt", "Vaelindrath", "Highmere Stair", "nine crowns", "meteoric",
    "Emberfall", "ledger.txt", "highmere.txt", "emberfall.txt", "Test Corpus",
    "c1.txt", "c2.txt", "c3.txt",
]


def _build(corpus: Path, out: Path, **kw) -> PackageResult:
    kw.setdefault("config", FAST)
    kw.setdefault("name", "Test Corpus")
    return build_package(corpus, out, **kw)


def _extract(archive: Path, dest: Path) -> Path:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest, filter="data")
    roots = [p for p in dest.iterdir() if p.is_dir()]
    assert len(roots) == 1
    return roots[0]


def _run(root: Path, *args: str, env: dict | None = None, timeout: int = 120):
    full = {**os.environ, "PYTHONPATH": ""}
    for k in ("RAG_PASSPHRASE", "RAG_KEYFILE", "RAG_HOST", "RAG_PORT", "RAG_TOP_K",
              "RAG_DEVICE", "RAG_RERANKER", "RAG_AUTH_TOKEN", "RAG_CONFIG"):
        full.pop(k, None)
    full.update(env or {})
    return subprocess.run(
        [sys.executable, str(root / "server.py"), *args],
        capture_output=True, text=True, timeout=timeout, check=False,
        env=full, stdin=subprocess.DEVNULL)


def _query(root: Path, q: str, env: dict | None = None) -> list[dict]:
    r = _run(root, "--corpus", str(root / "corpus"), "--query", q, "--top-k", "3", env=env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def _load_server():
    spec = importlib.util.spec_from_file_location(
        "standalone_server_under_test",
        Path(ec.__file__).with_name("standalone_server.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _all_bytes(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


# --------------------------------------------------------------- round trip
def test_roundtrip_matches_plaintext_baseline(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    plain = _extract(_build(corpus_dir, tmp_path / "p.tar.gz", encrypt=False).path,
                     tmp_path / "plain")
    res = _build(corpus_dir, tmp_path / "e.tar.gz", passphrase=PASS)
    assert res.encrypted and res.passphrase is None  # supplied -> not echoed back
    enc = _extract(res.path, tmp_path / "enc")
    assert (enc / "corpus" / "corpus.enc").is_file()
    assert not (enc / "corpus" / "chunks.jsonl").exists()
    assert (enc / "rag_container.py").is_file()
    assert "cryptography" in (enc / "requirements.txt").read_text()

    for q in QUERIES:
        base = _query(plain, q)
        got = _query(enc, q, env={"RAG_PASSPHRASE": PASS})
        assert got == base, q
    assert _query(enc, QUERIES[0], env={"RAG_PASSPHRASE": PASS})[0]["chunk_id"] == "c1"
    # sources are readable lazily through the container
    srv = _load_server()
    corpus = srv.Corpus(enc / "corpus", srv.resolve_config({}, {}, {})[0], PASS)
    assert corpus.read_source("c1.txt") == _TEXTS[0]


def test_keyfile_and_generated_passphrase(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    res = _build(corpus_dir, tmp_path / "g.tar.gz")  # no passphrase -> generated
    assert res.passphrase and len(res.passphrase) >= 20
    assert res.passphrase not in repr(res) and res.passphrase not in str(res)
    root = _extract(res.path, tmp_path / "x")
    kf = tmp_path / "pass.txt"
    kf.write_text(res.passphrase + "\n")
    hits = _run(root, "--corpus", str(root / "corpus"), "--query", "highmere crowns",
                "--keyfile", str(kf))
    assert hits.returncode == 0, hits.stderr
    assert json.loads(hits.stdout)[0]["chunk_id"] == "c2"
    # the generated key is not anywhere in the produced bytes
    needle = res.passphrase.encode()
    assert all(needle not in b for b in _all_bytes(root).values())
    assert needle not in res.path.read_bytes()


def test_passphrase_validation(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    with pytest.raises(ValueError, match="too short"):
        _build(corpus_dir, tmp_path / "a.tar.gz", passphrase="short")
    with pytest.raises(ValueError, match="encryption is disabled"):
        _build(corpus_dir, tmp_path / "b.tar.gz", passphrase=PASS, encrypt=False)


# ----------------------------------------------------- nothing in plaintext
@pytest.mark.parametrize("fmt", ["tar", "tar.gz", "zip"])
def test_archive_has_no_plaintext_content(corpus_dir: Path, tmp_path: Path, fmt: str) -> None:  # noqa: F811
    res = _build(corpus_dir, tmp_path / f"e.{fmt}", fmt=fmt, passphrase=PASS)
    blobs: dict[str, bytes] = {"<archive>": res.path.read_bytes()}
    root = _extract(res.path, tmp_path / "x")
    blobs.update(_all_bytes(root))
    if fmt == "tar":  # raw tar is uncompressed: the whole archive is scannable
        assert len(blobs["<archive>"]) > 1000
    for where, data in blobs.items():
        low = data.lower()
        for needle in NEEDLES:
            assert needle.lower().encode() not in low, f"{needle!r} leaked in {where}"
    # only the neutral root name, no project name in any member path
    names = [p.relative_to(tmp_path / "x").as_posix() for p in (tmp_path / "x").rglob("*")]
    assert all("test" not in n.lower() for n in names), names
    # header is minimal: version, KDF params, salt, nonce scheme
    enc = (root / "corpus" / "corpus.enc").read_bytes()
    assert enc.startswith(b"FTSRAGE1")
    hlen = int.from_bytes(enc[8:12], "big")
    header = json.loads(enc[12:12 + hlen])
    assert set(header) == {"v", "kdf", "n", "r", "p", "salt", "frame", "nonce", "prefix"}


def test_plaintext_opt_out_is_labelled(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    res = _build(corpus_dir, tmp_path / "p.tar.gz", encrypt=False)
    root = _extract(res.path, tmp_path / "x")
    assert not res.encrypted and root.name == "test-corpus-rag"
    assert "NOT ENCRYPTED" in (root / "README.md").read_text()
    assert not (root / "corpus" / "corpus.enc").exists()
    assert (root / "corpus" / "chunks.jsonl").is_file()


# ------------------------------------------------------ fail closed
def test_wrong_passphrase_fails_closed(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    root = _extract(_build(corpus_dir, tmp_path / "e.tar.gz", passphrase=PASS).path,
                    tmp_path / "x")
    r = _run(root, "--corpus", str(root / "corpus"), "--query", "oath",
             env={"RAG_PASSPHRASE": "definitely the wrong one"})
    assert r.returncode == 3
    assert r.stdout == ""
    assert "wrong passphrase" in r.stderr.lower()
    with pytest.raises(WrongPassphraseOrTampered):
        ContainerReader(root / "corpus" / "corpus.enc", "nope-nope-nope")
    # no passphrase at all (no tty / env): refuses, never guesses
    r2 = _run(root, "--corpus", str(root / "corpus"), "--query", "oath")
    assert r2.returncode == 2 and "passphrase" in r2.stderr


def test_flipped_byte_fails_everywhere(tmp_path: Path) -> None:
    path = tmp_path / "c.enc"
    big = os.urandom(150_000)  # > 2 frames
    with open(path, "wb") as fp:
        w = ContainerWriter(fp, PASS, log_n=10)
        w.add("a.bin", big)
        w.add("b.txt", b"hello world")
        w.close()
    blob = path.read_bytes()

    def read_all(data: bytes) -> None:
        p = tmp_path / "t.enc"
        p.write_bytes(data)
        with ContainerReader(p, PASS) as r:
            assert r.read("a.bin") == big and r.read("b.txt") == b"hello world"

    read_all(blob)  # control: untouched blob is fine
    positions = [14, 40, 200, 70_000, 140_000, len(blob) // 2, len(blob) - 100,
                 len(blob) - 30, len(blob) - 5]
    for pos in positions:
        bad = bytearray(blob)
        bad[pos] ^= 0x01
        with pytest.raises(ContainerError):
            read_all(bytes(bad))
    with pytest.raises(ContainerError):  # truncation
        read_all(blob[:-1])
    with pytest.raises(ContainerError):
        read_all(blob[:1000])


def test_flipped_byte_in_built_package_fails_via_cli(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    root = _extract(_build(corpus_dir, tmp_path / "e.tar.gz", passphrase=PASS).path,
                    tmp_path / "x")
    enc = root / "corpus" / "corpus.enc"
    data = bytearray(enc.read_bytes())
    data[len(data) // 2] ^= 0x80
    enc.write_bytes(bytes(data))
    r = _run(root, "--corpus", str(root / "corpus"), "--query", "oath",
             env={"RAG_PASSPHRASE": PASS})
    assert r.returncode == 3 and r.stdout == ""
    assert "tamper" in r.stderr.lower() or "corrupt" in r.stderr.lower()


# ------------------------------------------------------ temp-dir hygiene
def test_no_plaintext_left_in_tempdir(corpus_dir: Path, tmp_path: Path,  # noqa: F811
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setattr("tempfile.tempdir", str(tmpdir))
    before = sorted(p.name for p in tmpdir.rglob("*"))
    res = _build(corpus_dir, tmp_path / "out" / "e.tar.gz", passphrase=PASS)
    assert sorted(p.name for p in tmpdir.rglob("*")) == before, "build left files in TMPDIR"
    assert not list((tmp_path / "out").glob("*.part"))
    # a failing build also cleans up
    with pytest.raises(FileNotFoundError):
        _build(tmp_path / "missing", tmp_path / "out" / "bad.tar.gz")
    assert not list((tmp_path / "out").glob("*bad*"))

    root = _extract(res.path, tmp_path / "x")
    for env in ({"RAG_PASSPHRASE": PASS, "TMPDIR": str(tmpdir)},):
        r = _run(root, "--corpus", str(root / "corpus"), "--query", "oath of salt", env=env)
        assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in tmpdir.rglob("*")) == before, "server start left files"
    assert not any(p.suffix == ".part" for p in tmp_path.rglob("*"))
    # and the archive dir next to the output holds only the finished archive
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["e.tar.gz"]


# ------------------------------------------------------ config contract
def test_config_precedence_flag_env_file_default() -> None:
    srv = _load_server()
    cfg, origin = srv.resolve_config(
        {"port": 3333}, {"RAG_PORT": "2222", "RAG_TOP_K": "12"},
        {"port": 1111, "top_k": 11, "device": "cuda"})
    assert (cfg["port"], origin["port"]) == (3333, "flag")
    assert (cfg["top_k"], origin["top_k"]) == (12, "env")
    assert (cfg["device"], origin["device"]) == ("cuda", "file")
    assert (cfg["host"], origin["host"]) == ("127.0.0.1", "default")
    assert cfg["reranker"] is True
    # legacy env names keep working; empty env does not mask the file
    cfg, origin = srv.resolve_config(
        {}, {"RAG_EMBED_MODEL": "m", "RAG_EMBED_BASE_URL": "http://x/v1",
             "RAG_EMBED_API_KEY": "k", "RAG_DEVICE": ""}, {"device": "mps"})
    assert (cfg["embed_model"], cfg["embed_base_url"], cfg["embed_api_key"]) == ("m", "http://x/v1", "k")
    assert cfg["device"] == "mps"
    with pytest.raises(srv.ConfigError):
        srv.resolve_config({"port": 70000}, {}, {})


def test_print_config_merges_and_redacts(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    cfgx = RagExportConfig(kdf_log_n=10, port=9001, top_k=9, device="cuda:1",
                           reranker_enabled=False)
    root = _extract(_build(corpus_dir, tmp_path / "e.tar.gz", config=cfgx,
                           passphrase=PASS).path, tmp_path / "x")
    shipped = json.loads((root / "rag.config.json").read_text())
    assert shipped == {"device": "cuda:1", "host": "127.0.0.1", "port": 9001,
                       "top_k": 9, "reranker": False}
    assert "passphrase" not in json.dumps(shipped).lower()
    r = _run(root, "--print-config", "--top-k", "4",
             env={"RAG_PORT": "9100", "RAG_AUTH_TOKEN": "tok-SECRET-123",
                  "RAG_EMBED_API_KEY": "key-SECRET-456"})
    assert r.returncode == 0, r.stderr
    assert "SECRET" not in r.stdout
    out = json.loads(r.stdout)
    assert out["encrypted"] is True
    c, o = out["config"], out["origin"]
    assert (c["top_k"], o["top_k"]) == (4, "flag")
    assert (c["port"], o["port"]) == (9100, "env")
    assert (c["device"], o["device"], c["reranker"]) == ("cuda:1", "file", False)
    assert c["host"] == "127.0.0.1" and c["auth_token"] == "***"


def test_scripts_pass_flags_through(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    root = _extract(_build(corpus_dir, tmp_path / "e.tar.gz", passphrase=PASS).path,
                    tmp_path / "x")
    for sh in ("install.sh", "run-http.sh", "run-mcp.sh", "setup.sh"):
        assert subprocess.run(["bash", "-n", str(root / sh)], check=False).returncode == 0
        assert (root / sh).stat().st_mode & 0o111
    # real exec of the run scripts with a stub venv -> flags reach server.py
    (root / ".venv" / "bin").mkdir(parents=True)
    stub = root / ".venv" / "bin" / "python"  # wrapper keeps the real venv's site-packages
    stub.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    stub.chmod(0o755)
    for sh in ("run-http.sh", "run-mcp.sh"):
        r = subprocess.run(
            ["bash", str(root / sh), "--print-config", "--port", "9123", "--top-k", "7",
             "--device", "cuda", "--no-reranker", "--host", "127.0.0.1"],
            capture_output=True, text=True, timeout=60, check=False, cwd=str(root),
            env={**os.environ, "PYTHONPATH": ""})
        assert r.returncode == 0, r.stderr
        c = json.loads(r.stdout)["config"]
        assert (c["port"], c["top_k"], c["device"], c["reranker"]) == (9123, 7, "cuda", False)
    # install.sh/setup.sh document & forward the same flags
    inst = (root / "install.sh").read_text()
    assert '--save-config "$@"' in inst and "--no-reranker" in inst
    setup = (root / "setup.sh").read_text()
    for flag in ("--config", "--device", "--host", "--top-k", "--no-reranker", "--keyfile"):
        assert flag in setup


def test_non_loopback_bind_requires_token(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    srv = _load_server()
    for host in ("0.0.0.0", "192.168.1.5", "::", "example.com"):
        with pytest.raises(srv.ConfigError, match="auth token"):
            srv.check_bind_security({"host": host, "auth_token": ""})
        srv.check_bind_security({"host": host, "auth_token": "t"})
    for host in ("127.0.0.1", "localhost", "::1", "127.0.0.5"):
        srv.check_bind_security({"host": host, "auth_token": ""})

    root = _extract(_build(corpus_dir, tmp_path / "e.tar.gz", encrypt=False).path,
                    tmp_path / "x")
    r = _run(root, "--corpus", str(root / "corpus"), "--http", "--host", "0.0.0.0",
             "--port", "18999", timeout=30)
    assert r.returncode == 2 and "auth token" in r.stderr
    # file/env host without token is refused too, and before any passphrase prompt
    enc = _extract(_build(corpus_dir, tmp_path / "e2.tar.gz", passphrase=PASS).path,
                   tmp_path / "x2")
    r = _run(enc, "--corpus", str(enc / "corpus"), "--http", env={"RAG_HOST": "0.0.0.0"}, timeout=30)
    assert r.returncode == 2 and "auth token" in r.stderr and "passphrase" not in r.stderr


def test_http_bearer_auth_and_no_cors(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    srv = _load_server()
    root = _extract(_build(corpus_dir, tmp_path / "e.tar.gz", passphrase=PASS).path,
                    tmp_path / "x")
    corpus = srv.Corpus(root / "corpus", srv.resolve_config({}, {}, {})[0], PASS)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(corpus, "s3cret-token"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/search?q=oath", timeout=10)
        assert e.value.code == 401
        bad = urllib.request.Request(base + "/search?q=oath",
                                     headers={"Authorization": "Bearer nope"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad, timeout=10)
        assert e.value.code == 401
        ok = urllib.request.Request(base + "/search?q=oath+of+salt",
                                    headers={"Authorization": "Bearer s3cret-token"})
        with urllib.request.urlopen(ok, timeout=10) as r:
            assert r.status == 200
            assert r.headers.get("Access-Control-Allow-Origin") is None
            body = json.load(io.TextIOWrapper(r))
        assert body["hits"][0]["chunk_id"] == "c1"
    finally:
        httpd.shutdown()


# ------------------------------------------------------ export-config model
def test_export_config_defaults_and_validation() -> None:
    d = RagExportConfig()
    assert d.encrypt and d.host == "127.0.0.1" and d.port == 8899 and d.top_k == 10
    assert d.archive_format == "tar.gz" and not d.include_models and d.device == "auto"
    for bad in ({"device": "tpu"}, {"port": 0}, {"top_k": 0}, {"archive_format": "rar"},
                {"host": "a b"}, {"kdf_log_n": 5}):
        with pytest.raises(ValueError):
            RagExportConfig(**bad)


def test_export_config_layers_and_no_secret_persisted(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.webui.routes import settings as s
    sp = tmp_path / "settings.json"
    sp.write_text(json.dumps({"theme": "keep-me"}))
    monkeypatch.setattr(s, "SETTINGS_PATH", sp)
    assert ec.load_defaults() == RagExportConfig()
    ec.save_defaults({"port": 9000, "include_models": True})
    ec.save_project_override("p1", {"port": 9100, "encrypt": False})
    eff = ec.effective_config("p1", {"top_k": 8, "device": None})
    assert (eff.port, eff.include_models, eff.encrypt, eff.top_k) == (9100, True, False, 8)
    assert ec.effective_config("p2").port == 9000
    ec.save_project_override("p1", None)
    assert ec.effective_config("p1").port == 9000
    data = json.loads(sp.read_text())
    assert data["theme"] == "keep-me"
    assert "pass" not in json.dumps(data).lower() and "key" not in json.dumps(data["rag_export"]).lower()
    sp.write_text(json.dumps({"rag_export": {"port": "garbage", "top_k": 3, "bogus": 1}}))
    assert ec.load_defaults().top_k == 3 and ec.load_defaults().port == 8899  # stale keys ignored


def test_build_uses_config_models_and_zip(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    res = _build(corpus_dir, tmp_path / "z.zip", config=RagExportConfig(
        kdf_log_n=10, archive_format="zip", host="127.0.0.1", port=9555), passphrase=PASS)
    assert res.path.suffix == ".zip"
    root = _extract(res.path, tmp_path / "x")
    assert json.loads((root / "rag.config.json").read_text())["port"] == 9555
    assert _query(root, "highmere crowns", env={"RAG_PASSPHRASE": PASS})[0]["chunk_id"] == "c2"
    assert _IDS  # fixture sanity
