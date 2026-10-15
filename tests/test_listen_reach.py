"""Who can open the web UI (Genor 2026-10-15): switches for this computer (always), the local network and Tailscale,
plus custom addresses. Bound as narrowly as the switches allow; a connection arriving on a switched-off address is
refused at once; anything beyond this computer needs a password; Tailscale is detected and offered."""
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

from finetune_studio.access import install_config, netinfo, server, store, tailscale
from finetune_studio.access import middleware as mw
from finetune_studio.access.middleware import AccessMiddleware

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


# ── the model ───────────────────────────────────────────────────────────

def _ts_on(**kw) -> server.ServerAddress:
    return server.change(server.make(7860, **kw), tailscale=True, tailnet=TAILNET)


def test_binds_are_as_narrow_as_the_switches() -> None:
    assert server.make().binds() == ("127.0.0.1",)
    assert _ts_on().binds() == ("127.0.0.1", TS_IP)                  # the LAN cannot even connect
    assert server.make(custom=["198.51.100.2"]).binds() == ("127.0.0.1", "198.51.100.2")
    assert server.make(lan=True).binds() == ("0.0.0.0",)              # DHCP moves LAN addresses: wildcard + filter
    assert _ts_on(lan=True).binds() == ("0.0.0.0",)
    assert server.make(custom=["::"]).binds() == ("::",)
    assert server.make(lan=True, custom=["fd00::5"]).binds() == ("0.0.0.0", "fd00::5")


def test_each_switch_decides_per_arrival_address() -> None:
    lan_only = server.make(lan=True)
    assert lan_only.allows("127.0.0.1") and lan_only.allows("192.0.2.7") and not lan_only.allows(TS_IP)
    ts_only = _ts_on()
    assert ts_only.allows(TS_IP) and not ts_only.allows("192.0.2.7") and ts_only.allows("::ffff:127.0.0.1")
    custom = server.make(custom=["198.51.100.2"])
    assert custom.allows("198.51.100.2") and not custom.allows("198.51.100.3")
    assert server.make(custom=["0.0.0.0"]).allows(TS_IP) and server.make(custom=["0.0.0.0"]).everything


def test_descriptions_and_password_need() -> None:
    assert server.describe(server.make()) == "this computer only" and server.make().loopback_only
    assert server.describe(_ts_on(lan=True)) == "this computer + local network + Tailscale"
    assert server.describe(server.make(custom=["198.51.100.2", "198.51.100.3"])) == "this computer + 2 custom addresses"
    assert server.describe(server.make(custom=["0.0.0.0"])) == "every network"
    assert not _ts_on().loopback_only and not server.make(lan=True).loopback_only
    assert server.make(custom=["127.0.0.2"]).loopback_only


def test_bad_values_are_refused_with_the_reason() -> None:
    with pytest.raises(ValueError, match="IP addresses"):
        server.make(custom=["box.example"])
    with pytest.raises(ValueError, match="port"):
        server.make(70000)
    with pytest.raises(ValueError, match="not detected"):
        server.make(tailscale=True)
    with pytest.raises(ValueError, match="not a Tailscale address"):
        server.make(tailscale=True, tailscale_ip="192.0.2.5")
    assert server.make(lan=True, tailscale=True).binds() == ("0.0.0.0",)   # wildcard: no Tailscale IP needed


def test_server_json_roundtrip_and_old_single_host_files(tmp_path) -> None:
    addr = _ts_on(lan=True, custom=["198.51.100.2"])
    server.save(addr)
    assert server.load() == addr
    f = server.server_file()
    f.write_text(json.dumps({"host": "0.0.0.0", "port": 7861, "allowed_hosts": []}))
    old = server.load()
    assert old.lan and old.tailscale and old.port == 7861 and old.binds() == ("0.0.0.0",)
    f.write_text(json.dumps({"host": "127.0.0.1", "port": 7860, "tailscale": True, "tailscale_ip": TS_IP,
                             "tailscale_names": ["box"]}))
    assert server.load().binds() == ("127.0.0.1", TS_IP)
    f.write_text(json.dumps({"host": "198.51.100.2", "port": 7860}))
    assert server.load().custom == ("198.51.100.2",)


def test_command_line_host_is_the_old_form_and_keeps_tailscale_names() -> None:
    server.save(_ts_on())
    assert server.resolve(None, 9000).binds() == ("127.0.0.1", TS_IP) and server.resolve(None, 9000).port == 9000
    legacy = server.resolve("0.0.0.0", None)
    assert legacy.lan and legacy.tailscale_ip == TS_IP


def test_the_start_follows_a_changed_tailscale_address_and_keeps_the_last_one_while_down(monkeypatch) -> None:
    server.save(_ts_on())
    moved = tailscale.Tailnet(ipv4=TS_IP_2, dns_name="box.tail1234.ts.net")
    monkeypatch.setattr(tailscale, "detect", lambda timeout=3.0: moved)
    assert server.refresh_tailscale(server.load()).binds() == ("127.0.0.1", TS_IP_2)
    assert server.load().tailscale_ip == TS_IP_2, "the new address is saved back"
    monkeypatch.setattr(tailscale, "detect", lambda timeout=3.0: None)
    assert server.refresh_tailscale(server.load()).binds() == ("127.0.0.1", TS_IP_2)


def test_effective_reach_is_what_is_bound_filtered_by_the_switches() -> None:
    assert server.effective(("127.0.0.1",), server.make(lan=True)) == ["this computer"]       # not applied yet
    assert server.effective(("0.0.0.0",), server.make()) == ["this computer"]                 # switched off: refused
    assert server.effective(("0.0.0.0",), _ts_on(lan=True)) == ["this computer", "local network", "Tailscale"]
    assert server.effective(("127.0.0.1", TS_IP), _ts_on()) == ["this computer", "Tailscale"]


def test_lan_addresses_skip_containers_vpns_and_tailscale(monkeypatch) -> None:
    import psutil
    a = lambda ip: SimpleNamespace(address=ip)
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: {
        "lo": [a("127.0.0.1")], "enp3s0": [a("192.0.2.10"), a("aa:bb:cc:dd:ee:ff"), a("fe80::1%enp3s0")],
        "docker0": [a("172.17.0.1")], "tailscale0": [a(TS_IP)], "wlan0": [a("192.0.2.11")], "eth9": [a("192.0.2.99")]})
    monkeypatch.setattr(psutil, "net_if_stats", lambda: {"eth9": SimpleNamespace(isup=False),
                                                         "enp3s0": SimpleNamespace(isup=True)})
    assert [(x.interface, x.ip) for x in netinfo.lan_addresses()] == [("enp3s0", "192.0.2.10"), ("wlan0", "192.0.2.11")]


# ── middleware ──────────────────────────────────────────────────────────

async def _call(addr_local: str, host: str = "localhost") -> tuple[int, bytes]:
    sent: list[dict] = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "path": "/api/projects", "method": "GET", "server": (addr_local, 7860),
             "headers": [(b"host", host.encode())], "query_string": b""}
    await AccessMiddleware(app)(scope, None, send)
    return sent[0]["status"], b"".join(m.get("body", b"") for m in sent[1:])


@pytest.mark.asyncio
async def test_a_switched_off_address_is_refused_at_once() -> None:
    server.save(server.make(lan=True))
    assert (await _call("192.0.2.7", "192.0.2.7:7860"))[0] == 200
    status, body = await _call(TS_IP, f"{TS_IP}:7860")
    assert status == 403 and b"from Tailscale is switched off" in body
    server.save(server.make())                                   # LAN off, no restart: refused right away
    status, body = await _call("192.0.2.7", "192.0.2.7:7860")
    assert status == 403 and b"from the local network is switched off" in body
    assert (await _call("127.0.0.1"))[0] == 200


def test_tailscale_host_names_are_accepted_only_while_it_is_on(client) -> None:
    for host in (f"{TS_IP}:7860", "box.tail1234.ts.net:7860", "box:7860"):
        assert client.get("/api/projects", headers={"Host": host}).status_code == 403, host
    server.save(_ts_on())
    for host in (f"{TS_IP}:7860", "box.tail1234.ts.net:7860", "box:7860"):
        assert client.get("/api/projects", headers={"Host": host}).status_code == 200, host
    assert client.get("/api/projects", headers={"Host": "attacker.example"}).status_code == 403
    server.save(server.make(lan=True))                           # LAN on: LAN names vary, no host check
    assert client.get("/api/projects", headers={"Host": "anything.lan"}).status_code == 200


def test_a_tailscale_listener_without_a_password_is_reported(client, monkeypatch) -> None:
    server.save(_ts_on())
    monkeypatch.setenv("FTS_LISTEN_ADDRS", f"127.0.0.1,{TS_IP}")
    assert mw.listening() == ("127.0.0.1", TS_IP)
    assert client.get("/api/auth/state").json()["exposed_without_password"] is True
    store.set_password(PW)
    assert client.get("/api/auth/state").json()["exposed_without_password"] is False


# ── Settings -> Access & network ────────────────────────────────────────

def test_settings_shows_each_switch(client, on_tailnet, monkeypatch) -> None:
    monkeypatch.setattr(netinfo, "lan_addresses", lambda: [netinfo.LanAddress("enp3s0", "192.0.2.10")])
    st = client.get("/api/settings/security").json()
    r = st["reach"]
    assert r["tailscale"]["detected"]["ip"] == TS_IP and r["tailscale"]["on"] is False
    assert r["lan"] == {"on": False, "listening": False, "addresses": [{"interface": "enp3s0", "ip": "192.0.2.10"}]}
    assert r["now"] == ["this computer"] and r["saved"] == "this computer only"
    assert st["server"]["binds"] == ["127.0.0.1"] and st["server"]["restart_needed"] is False


def test_switches_need_a_password_and_on_waits_for_a_restart(client, on_tailnet, monkeypatch) -> None:
    monkeypatch.setenv("FTS_LISTEN_ADDRS", "127.0.0.1")                  # what the supervisor bound
    body = {"tailscale": True}
    r = client.put("/api/settings/security/server", json=body, headers=SAME)
    assert r.status_code == 409 and "Tailscale" in r.json()["detail"] and not server.load().tailscale
    store.set_password(PW)
    client.post("/api/auth/login", json={"password": PW}, headers=SAME)
    r = client.put("/api/settings/security/server", json=body, headers=SAME)
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["reach"]["tailscale"]["on"] is True and st["server"]["restart_needed"] is True
    assert server.load().binds() == ("127.0.0.1", TS_IP)
    # Fields not sent are kept; the port and LAN switch change on their own.
    r = client.put("/api/settings/security/server", json={"port": 7861, "lan": True}, headers=SAME)
    assert r.status_code == 200 and server.load().tailscale and server.load().lan and server.load().port == 7861
    r = client.put("/api/settings/security/server", json={"custom": ["box.example"]}, headers=SAME)
    assert r.status_code == 422 and "IP addresses" in r.json()["detail"]
    r = client.put("/api/settings/security/server", json={"host": "127.0.0.1"}, headers=SAME)   # the old form
    assert r.status_code == 200 and server.load().loopback_only


def test_turning_tailscale_on_when_it_is_not_running_is_a_409(client, off_tailnet) -> None:
    r = client.put("/api/settings/security/server", json={"tailscale": True, "confirm_no_password": True},
                   headers=SAME)
    assert r.status_code == 409 and "not detected" in r.json()["detail"]


def test_settings_page_has_the_switches_and_apply_now(client) -> None:
    html = client.get("/settings").text
    for el in ('id="access-lan"', 'id="access-ts"', 'id="access-custom"', 'id="access-apply"', "security.js?v=3"):
        assert el in html, el
    js = (Path(__file__).parents[1] / "src/finetune_studio/webui/static/js/security.js").read_text()
    assert "renderReach" in js and "/api/service/components/web/restart" in js and "no-cors" in js


# ── install + CLI ───────────────────────────────────────────────────────

def test_install_config_switches_with_a_password(tmp_path: Path, on_tailnet) -> None:
    pw = tmp_path / "pw"
    pw.write_text(PW + "\n")
    cfg = tmp_path / "access.env"
    cfg.write_text(f"FTS_AUTH=password\nFTS_ADMIN_PASSWORD_FILE={pw}\nFTS_TAILSCALE=1\nFTS_LISTEN_CUSTOM=198.51.100.2\n")
    done = install_config.apply(install_config.parse(cfg))
    assert any("Tailscale" in line for line in done)
    assert server.load().binds() == ("127.0.0.1", TS_IP, "198.51.100.2") and store.load().enabled


def test_install_config_refuses_switches_without_a_password_or_a_tailnet(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "access.env"
    cfg.write_text("FTS_TAILSCALE=1\n")
    monkeypatch.setattr(tailscale, "detect", lambda timeout=3.0: TAILNET)
    with pytest.raises(ValueError, match="Tailscale can reach this instance but FTS_AUTH is not 'password'"):
        install_config.apply(install_config.parse(cfg))
    monkeypatch.setattr(tailscale, "detect", lambda timeout=3.0: None)
    with pytest.raises(ValueError, match="not detected"):
        install_config.apply(install_config.parse(cfg))
    cfg.write_text("FTS_LISTEN_LAN=maybe\n")
    with pytest.raises(ValueError, match="1 or 0"):
        install_config.parse(cfg)


def test_the_install_questions_ask_per_switch(monkeypatch, on_tailnet, capsys) -> None:
    answers = iter(["n", "y"])                                    # LAN no, Tailscale yes
    monkeypatch.setattr("builtins.input", lambda prompt="": print(prompt) or next(answers))
    monkeypatch.setattr(install_config.getpass, "getpass", lambda prompt="": PW)
    install_config.interactive()
    out = capsys.readouterr().out
    assert "This computer always can" in out and "Tailscale devices (Tailscale detected" in out
    addr = server.load()
    assert addr.binds() == ("127.0.0.1", TS_IP) and not addr.lan and store.load().enabled


def test_fts_server_set_switches_and_show(on_tailnet, capsys, monkeypatch) -> None:
    from finetune_studio.cli.commands.access import cmd_server
    monkeypatch.setattr(netinfo, "lan_addresses", lambda: [netinfo.LanAddress("enp3s0", "192.0.2.10")])
    store.set_password(PW)
    base = {"server_action": "set", "host": None, "port": None, "allowed_hosts": None, "yes": False,
            "lan": None, "tailscale": None, "custom": None}
    cmd_server(SimpleNamespace(**{**base, "tailscale": "on"}))
    cmd_server(SimpleNamespace(**{**base, "port": 7900}))
    cmd_server(SimpleNamespace(**{**base, "lan": "on", "custom": "198.51.100.2"}))
    addr = server.load()
    assert addr.port == 7900 and addr.tailscale and addr.lan and addr.custom == ("198.51.100.2",)
    capsys.readouterr()
    cmd_server(SimpleNamespace(server_action="show"))
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "127.0.0.1:7900 (this computer + local network + Tailscale + 1 custom address)"
    assert any("local network   on" in line and "192.0.2.10 enp3s0" in line for line in out)


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
        assert spec.env()["FTS_LISTEN_ADDRS"] == "127.0.0.1,127.0.0.2"
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
            assert spec.env()["FTS_LISTEN_ADDRS"] == "127.0.0.1"
        finally:
            spec.release()


def test_a_changed_plan_rebinds_on_the_next_web_start_and_a_busy_port_keeps_the_old_one() -> None:
    from finetune_studio.supervisor.web import web_spec
    first, second = _free_port(), _free_port()
    plan = {"v": (("127.0.0.1",), first)}
    spec = web_spec(plan=lambda: plan["v"])
    try:
        spec.prepare()
        assert spec.env()["FTS_LISTEN_PORT"] == str(first)
        plan["v"] = (("127.0.0.1",), second)                     # Settings saved a new port; fts restart web
        spec.prepare()
        assert spec.env()["FTS_LISTEN_PORT"] == str(second) and spec.env()["FTS_LISTEN_ERROR"] == ""
        with socket.create_connection(("127.0.0.1", second), timeout=2):
            pass
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", first), timeout=2).close()
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen(1)
            plan["v"] = (("127.0.0.1",), taken.getsockname()[1])
            spec.prepare()                                       # busy: the UI must stay reachable
            env = spec.env()
            assert env["FTS_LISTEN_PORT"] == str(second) and "Address already in use" in env["FTS_LISTEN_ERROR"]
            with socket.create_connection(("127.0.0.1", second), timeout=2):
                pass
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
