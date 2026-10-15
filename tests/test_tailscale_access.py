"""Tailscale listening (Genor 2026-10-15): detect the tailnet, offer "also listen there", bind a second socket on the
Tailscale IP only (the LAN stays out), accept its host names, and count it as network exposure (password)."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from finetune_studio.access import install_config, server, store, tailscale
from finetune_studio.access import middleware as mw

PW = "correct horse battery"
SAME = {"Origin": "http://testserver"}
# Built from parts: tests/test_no_private_hosts.py rejects literal tailnet addresses in the tree.
TS_IP = "100." "101.102.103"
TS_IP_2 = "100." "64.0.9"
TAILNET = tailscale.Tailnet(ipv4=TS_IP, dns_name="box.tail1234.ts.net", hostname="box")

STATUS_JSON = json.dumps({
    "BackendState": "Running",
    "Self": {"TailscaleIPs": ["fd7a:115c:a1e0::1", TS_IP], "DNSName": "Box.tail1234.ts.net.", "HostName": "Box"},
})


@pytest.fixture
def on_tailnet(monkeypatch: pytest.MonkeyPatch) -> tailscale.Tailnet:
    monkeypatch.setattr(tailscale, "detect", lambda timeout=3.0: TAILNET)
    monkeypatch.setattr(tailscale, "detect_cached", lambda max_age_s=30.0: TAILNET)
    return TAILNET


@pytest.fixture
def off_tailnet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tailscale, "detect", lambda timeout=3.0: None)
    monkeypatch.setattr(tailscale, "detect_cached", lambda max_age_s=30.0: None)


# ── detection ───────────────────────────────────────────────────────────

def test_status_json_gives_the_ipv4_and_every_host_name() -> None:
    t = tailscale.parse_status(STATUS_JSON)
    assert t == tailscale.Tailnet(ipv4=TS_IP, dns_name="box.tail1234.ts.net", hostname="box", source="cli")
    assert t.host_names() == (TS_IP, "box", "box.tail1234.ts.net")


def test_a_logged_out_or_odd_status_is_not_a_tailnet() -> None:
    assert tailscale.parse_status(json.dumps({"BackendState": "NeedsLogin", "Self": {"TailscaleIPs": [TS_IP]}})) is None
    assert tailscale.parse_status(json.dumps({"BackendState": "Running", "Self": {"TailscaleIPs": ["192.0.2.1"]}})) is None
    assert tailscale.parse_status("not json") is None


def test_with_the_cli_installed_its_answer_is_final(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tailscale.shutil, "which", lambda name: "/usr/bin/tailscale")
    monkeypatch.setattr(tailscale.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="not running"))
    monkeypatch.setattr(tailscale, "_from_interfaces", lambda: pytest.fail("a stale tailscale0 address must not count"))
    assert tailscale.detect() is None
    monkeypatch.setattr(tailscale.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=STATUS_JSON))
    assert tailscale.detect().ipv4 == TS_IP


def test_without_the_cli_an_interface_in_the_tailnet_range_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil
    monkeypatch.setattr(tailscale.shutil, "which", lambda name: None)
    def addr(a: str) -> SimpleNamespace:
        return SimpleNamespace(address=a)
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: {"eth0": [addr("192.0.2.5")], "tailscale0": [addr(TS_IP)]})
    assert tailscale.detect() == tailscale.Tailnet(ipv4=TS_IP, source="interface")
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: {"eth0": [addr("192.0.2.5")]})
    assert tailscale.detect() is None


# ── server.json ─────────────────────────────────────────────────────────

def test_server_json_keeps_tailscale_and_stays_short_without_it() -> None:
    plain = server.validate("127.0.0.1", 7860)
    server.save(plain)
    assert "tailscale" not in server.server_file().read_text()
    assert server.load() == plain
    on = server.with_tailscale(plain, True, TAILNET)
    server.save(on)
    loaded = server.load()
    assert loaded == on and loaded.tailscale and loaded.tailscale_ip == TS_IP
    assert loaded.tailscale_names == TAILNET.host_names()
    assert not loaded.loopback_only, "Tailscale devices are other machines: it needs a password like the network"
    assert loaded.extra_hosts() == (TS_IP,)
    assert server.describe(loaded) == f"this computer + Tailscale devices ({TS_IP})"


def test_every_interface_already_covers_tailscale() -> None:
    addr = server.with_tailscale(server.validate("0.0.0.0", 7860), True, TAILNET)
    assert addr.extra_hosts() == ()


def test_tailscale_needs_a_tailnet_address() -> None:
    with pytest.raises(ValueError, match="not detected"):
        server.validate("127.0.0.1", 7860, tailscale=True)
    with pytest.raises(ValueError, match="not a Tailscale address"):
        server.validate("127.0.0.1", 7860, tailscale=True, tailscale_ip="192.0.2.5")


def test_the_start_follows_a_changed_address_and_keeps_the_last_one_while_down(monkeypatch) -> None:
    server.save(server.with_tailscale(server.validate("127.0.0.1", 7860), True, TAILNET))
    moved = tailscale.Tailnet(ipv4=TS_IP_2, dns_name="box.tail1234.ts.net")
    monkeypatch.setattr(tailscale, "detect", lambda timeout=3.0: moved)
    assert server.refresh_tailscale(server.load()).extra_hosts() == (TS_IP_2,)
    assert server.load().tailscale_ip == TS_IP_2, "the new address is saved back"
    monkeypatch.setattr(tailscale, "detect", lambda timeout=3.0: None)
    assert server.refresh_tailscale(server.load()).extra_hosts() == (TS_IP_2,)


def test_command_line_host_keeps_the_saved_tailscale_choice() -> None:
    server.save(server.with_tailscale(server.validate("127.0.0.1", 7860), True, TAILNET))
    addr = server.resolve(None, 9000)
    assert addr.port == 9000 and addr.extra_hosts() == (TS_IP,)


# ── middleware ──────────────────────────────────────────────────────────

def test_tailscale_host_names_are_accepted_only_while_listening_there(client) -> None:
    for host in (f"{TS_IP}:7860", "box.tail1234.ts.net:7860", "box:7860"):
        assert client.get("/api/projects", headers={"Host": host}).status_code == 403, host
    server.save(server.with_tailscale(server.validate("127.0.0.1", 7860), True, TAILNET))
    for host in (f"{TS_IP}:7860", "box.tail1234.ts.net:7860", "box:7860"):
        assert client.get("/api/projects", headers={"Host": host}).status_code == 200, host
    assert client.get("/api/projects", headers={"Host": "attacker.example"}).status_code == 403


def test_a_tailscale_listener_without_a_password_is_reported(client, monkeypatch) -> None:
    monkeypatch.setenv("FTS_LISTEN_HOST", "127.0.0.1")
    monkeypatch.setenv("FTS_LISTEN_EXTRA", TS_IP)
    assert mw.listen_extra() == (TS_IP,)
    assert client.get("/api/auth/state").json()["exposed_without_password"] is True
    store.set_password(PW)
    assert client.get("/api/auth/state").json()["exposed_without_password"] is False


# ── Settings -> Access & network ────────────────────────────────────────

def test_settings_shows_a_detected_tailnet(client, on_tailnet) -> None:
    ts = client.get("/api/settings/security").json()["tailscale"]
    assert ts["detected"]["ip"] == TS_IP and ts["detected"]["dns_name"] == "box.tail1234.ts.net"
    assert ts["enabled"] is False and ts["listening"] is False and ts["covered_by_host"] is False


def test_turning_tailscale_on_needs_a_password_or_confirmation(client, on_tailnet) -> None:
    body = {"host": "127.0.0.1", "port": 7860, "tailscale": True}
    r = client.put("/api/settings/security/server", json=body, headers=SAME)
    assert r.status_code == 409 and "Tailscale devices" in r.json()["detail"]
    assert not server.load().tailscale
    store.set_password(PW)
    client.post("/api/auth/login", json={"password": PW}, headers=SAME)
    r = client.put("/api/settings/security/server", json=body, headers=SAME)
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["tailscale"]["enabled"] is True and st["server"]["restart_needed"] is True
    assert server.load().extra_hosts() == (TS_IP,)
    # A save that does not mention Tailscale (older page, other field) leaves it on.
    r = client.put("/api/settings/security/server", json={"host": "127.0.0.1", "port": 7861}, headers=SAME)
    assert r.status_code == 200 and server.load().tailscale and server.load().port == 7861
    r = client.put("/api/settings/security/server", json={**body, "tailscale": False}, headers=SAME)
    assert r.status_code == 200 and not server.load().tailscale


def test_turning_tailscale_on_when_it_is_not_running_is_a_409(client, off_tailnet) -> None:
    r = client.put("/api/settings/security/server", json={"host": "127.0.0.1", "port": 7860, "tailscale": True,
                                                         "confirm_no_password": True}, headers=SAME)
    assert r.status_code == 409 and "not detected" in r.json()["detail"]


def test_settings_page_has_the_tailscale_toggle(client) -> None:
    html = client.get("/settings").text
    assert 'id="access-ts"' in html and "security.js?v=2" in html
    js = (Path(__file__).parents[1] / "src/finetune_studio/webui/static/js/security.js").read_text()
    assert "body.tailscale" in js and "renderTailscale" in js


# ── install + CLI ───────────────────────────────────────────────────────

def test_install_config_tailscale_with_a_password(tmp_path: Path, on_tailnet) -> None:
    pw = tmp_path / "pw"
    pw.write_text(PW + "\n")
    cfg = tmp_path / "access.env"
    cfg.write_text(f"FTS_AUTH=password\nFTS_ADMIN_PASSWORD_FILE={pw}\nFTS_TAILSCALE=1\n")
    done = install_config.apply(install_config.parse(cfg))
    assert any("Tailscale devices" in line for line in done)
    assert server.load().extra_hosts() == (TS_IP,) and store.load().enabled


def test_install_config_tailscale_is_refused_without_a_password_or_a_tailnet(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "access.env"
    cfg.write_text("FTS_TAILSCALE=1\n")
    monkeypatch.setattr(tailscale, "detect", lambda timeout=3.0: TAILNET)
    with pytest.raises(ValueError, match="Tailscale devices reach this instance but FTS_AUTH is not 'password'"):
        install_config.apply(install_config.parse(cfg))
    monkeypatch.setattr(tailscale, "detect", lambda timeout=3.0: None)
    with pytest.raises(ValueError, match="not detected"):
        install_config.apply(install_config.parse(cfg))
    cfg.write_text("FTS_TAILSCALE=maybe\n")
    with pytest.raises(ValueError, match="1 or 0"):
        install_config.parse(cfg)


def test_the_install_question_offers_tailscale_when_detected(monkeypatch, on_tailnet, capsys) -> None:
    answers = iter(["3"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(install_config.getpass, "getpass", lambda prompt="": PW)
    install_config.interactive()
    assert "3) This computer + my Tailscale devices" in capsys.readouterr().out
    addr = server.load()
    assert addr.host == "127.0.0.1" and addr.extra_hosts() == (TS_IP,) and store.load().enabled


def test_fts_server_set_keeps_tailscale_when_changing_the_port(on_tailnet, capsys) -> None:
    from finetune_studio.cli.commands.access import cmd_server
    store.set_password(PW)
    base = {"server_action": "set", "host": None, "port": None, "allowed_hosts": None, "yes": False}
    cmd_server(SimpleNamespace(**base, tailscale="on"))
    cmd_server(SimpleNamespace(**{**base, "port": 7900}, tailscale=None))
    addr = server.load()
    assert addr.port == 7900 and addr.extra_hosts() == (TS_IP,)
    assert "Tailscale devices" in capsys.readouterr().out


# ── the second socket, for real ─────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_web_spec_binds_the_extra_address_and_skips_one_it_cannot_bind() -> None:
    from finetune_studio.supervisor.web import web_spec
    port = _free_port()
    spec = web_spec("127.0.0.1", port, extra_hosts=("127.0.0.2",))
    try:
        spec.prepare()
        argv, fds = spec.argv(), spec.pass_fds()
        assert argv[2] == "finetune_studio.webui.serve" and argv.count("--fd") == 2 and len(fds) == 2
        assert spec.env()["FTS_LISTEN_EXTRA"] == "127.0.0.2"
    finally:
        spec.release()
    with socket.socket() as taken:                                   # the extra address is busy: skip, keep the UI
        taken.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        taken.bind(("127.0.0.3", port))
        taken.listen(1)
        spec = web_spec("127.0.0.1", port, extra_hosts=("127.0.0.3",))
        try:
            spec.prepare()
            assert spec.argv()[2] == "uvicorn" and len(spec.pass_fds()) == 1
            assert spec.env()["FTS_LISTEN_EXTRA"] == ""
        finally:
            spec.release()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="127.0.0.2 is loopback on Linux only")
def test_serve_answers_on_both_sockets(tmp_path: Path) -> None:
    from finetune_studio.supervisor.web import ListenSocket
    (tmp_path / "tiny_app.py").write_text(
        "async def app(scope, receive, send):\n"
        "    if scope['type'] != 'http':\n        return\n"
        "    await send({'type': 'http.response.start', 'status': 200, 'headers': []})\n"
        "    await send({'type': 'http.response.body', 'body': scope['server'][0].encode()})\n")
    port = _free_port()
    socks = [ListenSocket("127.0.0.1", port), ListenSocket("127.0.0.2", port, freebind=True)]
    fds = [s.fileno() for s in socks]
    proc = subprocess.Popen(
        [sys.executable, "-m", "finetune_studio.webui.serve", *(a for fd in fds for a in ("--fd", str(fd))),
         "--app", "tiny_app:app"],
        pass_fds=fds, cwd=tmp_path, env={**__import__("os").environ, "PYTHONPATH": str(tmp_path)},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        seen = {}
        deadline = time.monotonic() + 30
        for host in ("127.0.0.1", "127.0.0.2"):
            while host not in seen:
                try:
                    with urllib.request.urlopen(f"http://{host}:{port}/", timeout=2) as r:
                        seen[host] = r.read().decode()
                except OSError:
                    assert proc.poll() is None, proc.stdout.read().decode()
                    assert time.monotonic() < deadline, "serve did not answer"
                    time.sleep(0.2)
        assert seen == {"127.0.0.1": "127.0.0.1", "127.0.0.2": "127.0.0.2"}
    finally:
        proc.terminate()
        proc.wait(timeout=15)
        for s in socks:
            s.close()
