"""Access control (2026-10-15 review: the app listened on 0.0.0.0 with no sign-in; any web page could POST to it)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from finetune_studio.access import passwords, server, sessions, store, throttle
from finetune_studio.access import middleware as mw

PW = "correct horse battery"
SAME = {"Origin": "http://testserver"}


@pytest.fixture(autouse=True)
def _clean_throttle():
    throttle.reset()
    yield
    throttle.reset()


def _root() -> Path:
    return Path(os.environ["FTS_ROOT"])


# ── building blocks ─────────────────────────────────────────────────────

def test_password_hash_roundtrip_and_policy() -> None:
    h = passwords.hash_password(PW)
    assert h.startswith("scrypt$") and PW not in h
    assert passwords.verify_password(PW, h) and not passwords.verify_password(PW + "x", h)
    assert not passwords.verify_password(PW, "garbage")
    with pytest.raises(ValueError, match="at least"):
        passwords.hash_password("short")


def test_server_address_defaults_to_this_computer_only() -> None:
    assert server.load() == server.ServerAddress(7860) and server.load().binds() == ("127.0.0.1",)
    assert server.is_loopback("localhost") and server.is_loopback("::1") and server.is_loopback("127.0.0.2")
    assert not server.is_loopback("0.0.0.0") and not server.is_loopback("192.0.2.4") and not server.is_loopback("box")
    server.save(server.validate("0.0.0.0", 8080, ["Studio.Example"]))    # the pre-switches form
    assert server.load() == server.ServerAddress(8080, lan=True, tailscale=True, allowed_hosts=("studio.example",))
    with pytest.raises(ValueError):
        server.validate("0.0.0.0", 70000)


def test_auth_file_holds_hashes_only_and_is_private() -> None:
    store.set_password(PW)
    tok, plain = store.create_token("ci")
    f = _root() / "auth" / "auth.json"
    text = f.read_text()
    assert PW not in text and plain not in text
    assert oct(f.stat().st_mode & 0o777) == "0o600"
    assert store.verify_token(plain).id == tok.id
    assert store.verify_token(plain + "x") is None and store.verify_token("not-a-token") is None
    assert store.revoke_token(tok.id) and store.verify_token(plain) is None


def test_a_damaged_auth_file_fails_closed() -> None:
    store.set_password(PW)
    (_root() / "auth" / "auth.json").write_text("{not json")
    with pytest.raises(RuntimeError, match="unreadable"):
        store.load()


# ── no password (the default) ───────────────────────────────────────────

def test_without_a_password_everything_is_open_locally(client) -> None:
    assert client.get("/api/projects").status_code == 200
    assert client.get("/login", follow_redirects=False).status_code == 303     # nothing to sign in to


def test_cross_site_writes_are_refused_even_without_a_password(client) -> None:
    r = client.post("/api/projects", json={"name": "x"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403 and r.json()["code"] == "request_refused"
    r = client.post("/api/projects", json={"name": "x"}, headers={"Origin": "null"})
    assert r.status_code == 403
    assert client.post("/api/projects", json={"name": "ok"}, headers=SAME).status_code == 200
    assert client.post("/api/projects", json={"name": "cli"}).status_code == 200          # no Origin: not a browser


def test_dns_rebinding_host_is_refused_while_listening_locally(client) -> None:
    r = client.get("/api/projects", headers={"Host": "attacker.example"})
    assert r.status_code == 403 and "not allowed" in r.json()["detail"]
    server.save(server.validate("127.0.0.1", 7860, ["studio.example"]))
    assert client.get("/api/projects", headers={"Host": "studio.example:7860"}).status_code == 200
    assert client.get("/api/projects", headers={"Host": "localhost:7860"}).status_code == 200


def test_listening_on_the_network_without_a_password_is_reported(client, monkeypatch) -> None:
    monkeypatch.setenv("FTS_LISTEN_ADDRS", "0.0.0.0")
    assert client.get("/api/auth/state").json()["exposed_without_password"] is False   # LAN switched off: refused
    server.save(server.make(7860, lan=True))
    assert client.get("/api/auth/state").json()["exposed_without_password"] is True
    store.set_password(PW)
    assert client.get("/api/auth/state").json()["exposed_without_password"] is False


# ── with a password ─────────────────────────────────────────────────────

def _login(client, password: str = PW):
    return client.post("/api/auth/login", json={"password": password, "next": "/projects"}, headers=SAME)


def test_password_mode_asks_everyone_to_sign_in(client) -> None:
    store.set_password(PW)
    r = client.get("/api/projects")
    assert r.status_code == 401 and r.json()["code"] == "auth_required"
    r = client.get("/projects/abc/data?x=1", headers={"Accept": "text/html"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login?next=%2Fprojects%2Fabc%2Fdata%3Fx%3D1"
    assert client.get("/api/health").status_code == 200                       # the supervisor's probe
    assert client.get("/login").status_code == 200
    assert client.get("/static/css/login.css").status_code == 200


def test_sign_in_sets_a_session_cookie_and_sign_out_ends_it(client) -> None:
    store.set_password(PW)
    assert _login(client, "wrong password!").status_code == 401
    r = _login(client)
    assert r.status_code == 200 and r.json()["next"] == "/projects"
    cookie = r.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie.replace("Lax", "lax") and PW not in cookie
    assert client.get("/api/projects").status_code == 200
    assert client.post("/api/auth/logout", headers=SAME).status_code == 200
    assert client.get("/api/projects").status_code == 401


def test_cookie_writes_need_a_same_origin_header(client) -> None:
    store.set_password(PW)
    _login(client)
    r = client.post("/api/projects", json={"name": "no-origin"})
    assert r.status_code == 403 and r.json()["code"] == "csrf"
    assert client.post("/api/projects", json={"name": "same"}, headers=SAME).status_code == 200
    assert client.post("/api/projects", json={"name": "ref"},
                       headers={"Referer": "http://testserver/projects"}).status_code == 200


def test_api_tokens_work_without_a_cookie_and_bad_ones_are_refused(client) -> None:
    store.set_password(PW)
    _, plain = store.create_token("script")
    assert client.get("/api/projects", headers={"Authorization": f"Bearer {plain}"}).status_code == 200
    assert client.post("/api/projects", json={"name": "t"}, headers={"Authorization": f"Bearer {plain}"}).status_code == 200
    r = client.get("/api/projects", headers={"Authorization": "Bearer fts_nope"})
    assert r.status_code == 401 and r.json()["code"] == "auth_invalid_token"


def test_repeated_wrong_passwords_are_throttled(client) -> None:
    store.set_password(PW)
    for _ in range(throttle.FREE_TRIES):
        assert _login(client, "wrong password!").status_code == 401
    r = _login(client)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0


def test_next_only_goes_to_this_site(client) -> None:
    store.set_password(PW)
    for evil in ("//evil.example/x", "https://evil.example", "/\\evil.example"):
        r = client.post("/api/auth/login", json={"password": PW, "next": evil}, headers=SAME)
        assert r.json()["next"] == "/"
        throttle.reset()


# ── Settings -> Security ────────────────────────────────────────────────

def test_setting_a_password_from_settings_keeps_this_browser_signed_in(client) -> None:
    r = client.put("/api/settings/security/password", json={"new_password": PW}, headers=SAME)
    assert r.status_code == 200 and r.json()["password_set"] is True
    assert client.get("/api/projects").status_code == 200                     # fresh session cookie came back


def test_changing_the_password_needs_the_old_one_and_signs_out_other_browsers(client) -> None:
    store.set_password(PW)
    _login(client)
    other = sessions.create()
    r = client.put("/api/settings/security/password", json={"new_password": "another long one", "current_password": "x"},
                   headers=SAME)
    assert r.status_code == 403
    r = client.put("/api/settings/security/password",
                   json={"new_password": "another long one", "current_password": PW}, headers=SAME)
    assert r.status_code == 200 and not sessions.validate(other)
    assert client.get("/api/projects").status_code == 200


def test_weak_passwords_are_a_422_with_the_reason(client) -> None:
    r = client.put("/api/settings/security/password", json={"new_password": "short"}, headers=SAME)
    assert r.status_code == 422 and "at least" in r.json()["detail"]


def test_tokens_from_settings_are_shown_once(client) -> None:
    r = client.post("/api/settings/security/tokens", json={"name": "backup job"}, headers=SAME)
    assert r.status_code == 201 and r.json()["token"].startswith("fts_")
    listed = client.get("/api/settings/security").json()["tokens"]
    assert listed[0]["name"] == "backup job" and "token" not in listed[0] and "sha256" not in listed[0]
    assert client.delete(f"/api/settings/security/tokens/{listed[0]['id']}", headers=SAME).status_code == 204
    assert client.delete("/api/settings/security/tokens/nope", headers=SAME).status_code == 404


def test_opening_to_the_network_without_a_password_needs_confirmation(client) -> None:
    body = {"lan": True, "port": 7860}
    r = client.put("/api/settings/security/server", json=body, headers=SAME)
    assert r.status_code == 409 and "password" in r.json()["detail"] and "local network" in r.json()["detail"]
    r = client.put("/api/settings/security/server", json={**body, "confirm_no_password": True}, headers=SAME)
    assert r.status_code == 200 and r.json()["server"]["binds"] == ["0.0.0.0"]
    assert json.loads((_root() / "server.json").read_text())["lan"] is True


def test_disabling_the_password_needs_it(client) -> None:
    store.set_password(PW)
    _login(client)
    assert client.post("/api/settings/security/disable", json={"current_password": "nope nope"},
                       headers=SAME).status_code == 403
    r = client.post("/api/settings/security/disable", json={"current_password": PW}, headers=SAME)
    assert r.status_code == 200 and r.json()["mode"] == "off"


def test_the_middleware_rereads_auth_json_when_the_cli_changes_it(client) -> None:
    assert client.get("/api/projects").status_code == 200
    store.set_password(PW)                       # what `fts auth set-password` does while the app runs
    assert client.get("/api/projects").status_code == 401
    assert mw.exposed_without_password() is False


# ── install-time configuration (install.sh --config / the one question) ──

def _cfg(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "install.conf"
    p.write_text(text)
    return p


def test_install_config_network_with_password_file(tmp_path: Path) -> None:
    from finetune_studio.access import install_config
    (tmp_path / "pw").write_text(PW + "\n")
    cfg = install_config.parse(_cfg(tmp_path, f"""
        # automated install
        FTS_BIND_HOST=0.0.0.0
        FTS_PORT=7861
        FTS_AUTH=password
        FTS_ADMIN_PASSWORD_FILE={tmp_path / 'pw'}
        FTS_ALLOWED_HOSTS="studio.example, lab.example"
        FTS_INSTALL_SERVICE=1
    """))
    done = install_config.apply(cfg)
    assert cfg.install_service and any("password set" in d for d in done)
    assert server.load() == server.ServerAddress(7861, lan=True, tailscale=True,
                                                 allowed_hosts=("lab.example", "studio.example"))
    assert store.check_password(PW)


def test_install_config_refuses_the_network_without_a_password(tmp_path: Path) -> None:
    from finetune_studio.access import install_config
    with pytest.raises(ValueError, match="local network .*FTS_AUTH is not 'password'"):
        install_config.apply(install_config.parse(_cfg(tmp_path, "FTS_BIND_HOST=0.0.0.0\n")))
    assert not (_root() / "server.json").exists()                  # nothing half-written


def test_install_config_typos_are_errors_not_silently_open(tmp_path: Path) -> None:
    from finetune_studio.access import install_config
    with pytest.raises(ValueError, match="unknown key 'FTS_AUTHH'"):
        install_config.parse(_cfg(tmp_path, "FTS_AUTHH=password\n"))
    with pytest.raises(ValueError, match="needs FTS_ADMIN_PASSWORD_FILE"):
        install_config.parse(_cfg(tmp_path, "FTS_AUTH=password\n"))
    with pytest.raises(ValueError, match="'password' or 'off'"):
        install_config.parse(_cfg(tmp_path, "FTS_AUTH=yes\n"))


def test_cli_apply_config_reports_the_service_request(tmp_path: Path) -> None:
    import subprocess
    import sys
    conf = _cfg(tmp_path, f"FTS_AUTH=password\nFTS_ADMIN_PASSWORD={PW!r}\nFTS_INSTALL_SERVICE=1\n")
    r = subprocess.run([sys.executable, "-m", "finetune_studio.cli", "auth", "apply-config", str(conf)],
                       capture_output=True, text=True, env={**os.environ}, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "[access] password set" in r.stdout and r.stdout.strip().endswith("INSTALL_SERVICE=1")
    assert PW not in r.stdout + r.stderr
    bad = subprocess.run([sys.executable, "-m", "finetune_studio.cli", "auth", "apply-config", str(_cfg(tmp_path, "X=1\n"))],
                         capture_output=True, text=True, env={**os.environ}, timeout=60)
    assert bad.returncode == 2 and "unknown key" in bad.stderr
