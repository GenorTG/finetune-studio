"""`fts supervisor|status|start|stop|restart|logs|events|up|doctor` — talk to the supervisor.

These are the only CLI commands that work with the app down: ``doctor`` and ``up`` read the
machine, not the supervisor. Everything else needs the control socket and says so.
"""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from finetune_studio.supervisor.client import (
    SupervisorClient,
    SupervisorError,
    SupervisorUnavailable,
)
from finetune_studio.supervisor.paths import events_path, lock_path

UNIT = "finetune-studio"
EXIT_DOWN = 3  # supervisor unreachable
EXIT_DEGRADED = 1  # reachable, some component not ready


def _fail_down(exc: Exception) -> None:
    print(f"Supervisor is not running: {exc}\nStart it with `fts up` (manual or systemd); `fts doctor` explains why.",
          file=sys.stderr)
    sys.exit(EXIT_DOWN)


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(seconds)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def format_status(snap: dict[str, Any]) -> list[str]:
    sup = snap["supervisor"]
    lines = [f"supervisor pid {sup['pid']} up {_fmt_age(sup['uptime_s'])}", ""]
    lines.append(f"{'COMPONENT':<12}{'STATE':<11}{'PID':<9}{'UP':<9}{'RESTARTS':<9}LAST EXIT / HEALTH")
    for name, c in snap["components"].items():
        last = c.get("last_exit") or {}
        why = ("restarted on request" if last.get("reason") == "requested" else
               f"signal {last['signal']}" if last.get("signal") else f"code {last['code']}" if last.get("code") is not None
               else last.get("error", "")) if last else ""
        health = c.get("health", {}).get("detail", "")
        note = why or health
        if c["state"] == "backoff":
            note = f"retry in {c['retry_in_s']}s ({why})"
        lines.append(f"{name:<12}{c['state']:<11}{c['pid'] or '-':<9}{_fmt_age(c['uptime_s']):<9}{c['restarts']:<9}{note}")
    running = [j for j in snap.get("jobs", []) if j.get("running")]
    if running:
        lines += ["", f"{'JOB':<38}{'KIND':<11}{'PID':<9}{'UP':<9}TITLE"]
        for j in running:
            lines.append(f"{j['id']:<38}{j['kind']:<11}{j['pid']:<9}{_fmt_age(j['uptime_s']):<9}{j.get('title') or ''}")
    return lines


def cmd_status(args) -> None:
    try:
        snap = SupervisorClient().status()
    except SupervisorUnavailable as exc:
        _fail_down(exc)
    if args.json:
        print(json.dumps(snap, indent=2))
    else:
        print("\n".join(format_status(snap)))
    if any(c["state"] != "ready" for c in snap["components"].values()):
        sys.exit(EXIT_DEGRADED)


def cmd_component_action(args) -> None:
    try:
        SupervisorClient().act(args.name, args.command)
    except SupervisorUnavailable as exc:
        _fail_down(exc)
    except SupervisorError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"{args.command} requested for {args.name}")


def cmd_logs(args) -> None:
    try:
        lines = SupervisorClient().logs(args.name, args.lines)
    except SupervisorUnavailable as exc:
        _fail_down(exc)
    except SupervisorError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print("\n".join(lines))


def format_event(e: dict[str, Any]) -> str:
    """One event line; columns always separated (the 9-letter 'inference' used to run into the event kind)."""
    when = time.strftime("%H:%M:%S", time.localtime(e["at"]))
    extra = {k: v for k, v in e.items() if k not in ("seq", "at", "component", "kind", "tail")}
    return f"{when}  {e['component']:<10} {e['kind']:<18} {json.dumps(extra) if extra else ''}".rstrip()


def cmd_events(args) -> None:
    client = SupervisorClient()
    seen = 0
    try:
        while True:
            for e in client.events(seen, args.lines):
                seen = e["seq"]
                print(format_event(e))
            if not args.follow:
                return
            time.sleep(1)
    except SupervisorUnavailable as exc:
        _fail_down(exc)
    except KeyboardInterrupt:
        return


def cmd_supervisor(args) -> None:
    from finetune_studio.supervisor.__main__ import main
    argv = [*(["--host", args.host] if args.host else []), *(["--port", str(args.port)] if args.port else [])]
    sys.exit(main(argv))


def _unit_installed() -> bool:
    return (Path.home() / ".config/systemd/user" / f"{UNIT}.service").is_file()


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


# System tools the document parsers use (never pip packages): a missing one is a warning, its parse says what was skipped.
DOCUMENT_TOOLS = (
    ("tesseract", "OCR of scans and pictures", "apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-pol"),
    ("pdftoppm", "scanned PDF pages and page pictures", "apt install poppler-utils"),
    ("soffice|libreoffice", "files without a parser, .xls charts",
     "apt install libreoffice-core libreoffice-writer libreoffice-calc libreoffice-impress libreoffice-draw"),
)


def _unit_listen_host() -> str | None:
    """A unit written before server.json existed bakes ``--host`` into ExecStart; that one wins."""
    if not shutil.which("systemctl"):
        return None
    res = subprocess.run(["systemctl", "--user", "show", "-p", "ExecStart", "--value", UNIT],
                         capture_output=True, text=True, check=False)
    parts = res.stdout.replace(";", " ").split()
    return parts[parts.index("--host") + 1] if "--host" in parts[:-1] else None


def access_check() -> tuple[str, str, str]:
    from finetune_studio.access import server, store
    try:
        addr, cfg = server.load(), store.load()
    except (ValueError, RuntimeError) as e:
        return ("fail", "access", str(e))
    host = _unit_listen_host() or addr.host
    where = "this computer only" if server.is_loopback(host) else "the network"
    detail = f"listening on {host} ({where}), password {'on' if cfg.enabled else 'off'}, {len(cfg.tokens)} API token(s)"
    if not server.is_loopback(host) and not cfg.enabled:
        return ("fail", "access", detail + ": anyone on the network can control it -> fts auth set-password")
    return ("ok", "access", detail)


def doctor_checks(port: int | None = None) -> list[tuple[str, str, str]]:
    """(status, check, detail) rows; status is ok | warn | fail. Works with everything down."""
    from finetune_studio.access import server
    try:
        port = port or server.load().port
    except ValueError:
        port = port or 7860
    rows: list[tuple[str, str, str]] = []
    try:
        snap = SupervisorClient(timeout=2.0).status()
        rows.append(("ok", "supervisor", f"pid {snap['supervisor']['pid']}, {len(snap['components'])} component(s)"))
        for name, c in snap["components"].items():
            rows.append(("ok" if c["state"] == "ready" else "fail", f"component {name}", c["state"]))
    except SupervisorUnavailable as exc:
        rows.append(("fail", "supervisor", f"unreachable ({exc})"))
        rows.append(("warn", "lock file", f"{lock_path()} {'exists' if lock_path().exists() else 'absent'} (a stale lock is harmless)"))
    rows.append(("ok" if _port_open(port) else "fail", f"port {port}", "accepting connections" if _port_open(port) else "nothing listening"))
    if shutil.which("systemctl") and _unit_installed():
        res = subprocess.run(["systemctl", "--user", "is-active", UNIT], capture_output=True, text=True, check=False)
        state = res.stdout.strip() or "unknown"
        rows.append(("ok" if state == "active" else "fail", f"systemd unit {UNIT}", state))
    else:
        rows.append(("warn", f"systemd unit {UNIT}", "not installed (run install-service.sh)"))
    rows.append(access_check())
    from finetune_studio.accel.saved_choice import choice_path

    choice = choice_path()
    rows.append(("ok" if choice.is_file() else "warn", "compute device choice", str(choice) if choice.is_file() else "none saved (auto)"))
    for tool, use, hint in DOCUMENT_TOOLS:
        found = next((shutil.which(t) for t in tool.split("|") if shutil.which(t)), None)
        rows.append(("ok", f"tool {tool.split('|')[0]}", found) if found else ("warn", f"tool {tool.split('|')[0]}",
                                                                              f"missing: {use} ({hint})"))
    ev = events_path()
    if ev.is_file():
        last = ev.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:]
        if last:
            rows.append(("ok", "last supervisor event", last[0][:160]))
    return rows


def cmd_doctor(args) -> None:
    rows = doctor_checks(args.port)
    for status, check, detail in rows:
        print(f"[{status:>4}] {check:<26} {detail}")
    sys.exit(1 if any(r[0] == "fail" for r in rows) else 0)

