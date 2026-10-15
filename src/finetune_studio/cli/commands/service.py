"""`fts up|down|service` — start and stop the supervisor, with or without systemd.

Two independent ways to run the app, both first-class:
  * manual : ``fts up --manual`` (or ``fts supervisor`` in a terminal) starts a detached supervisor;
  * systemd: ``fts service install`` writes the user unit once, ``fts up --systemd`` starts it.
``fts up`` with no flag uses systemd when the unit is installed, else manual.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from finetune_studio.cli.commands.supervisor import UNIT, _unit_installed
from finetune_studio.supervisor.client import SupervisorClient, SupervisorUnavailable
from finetune_studio.supervisor.paths import run_dir, socket_path

START_WAIT_S = 90.0
STOP_WAIT_S = 45.0


def repo_root() -> Path | None:
    """The checkout this package runs from (editable install), or None for a wheel install."""
    root = Path(__file__).resolve().parents[4]
    return root if (root / "pyproject.toml").is_file() and (root / "install-service.sh").is_file() else None


def _systemctl(*args: str) -> int:
    return subprocess.run(["systemctl", "--user", *args], check=False).returncode


def _unit_active() -> bool:
    if not (shutil.which("systemctl") and _unit_installed()):
        return False
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", UNIT], check=False).returncode == 0


def _running() -> dict | None:
    try:
        return SupervisorClient(timeout=2.0).status()
    except SupervisorUnavailable:
        return None


def _wait_ready(deadline_s: float) -> dict | None:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        snap = _running()
        if snap and all(c["state"] == "ready" for c in snap["components"].values()):
            return snap
        time.sleep(0.5)
    return _running()


def _report_started(mode: str) -> None:
    snap = _wait_ready(START_WAIT_S)
    if snap is None:
        log = run_dir() / "supervisor.log"
        print(f"Started ({mode}) but the supervisor does not answer yet. Run `fts doctor`." + (f" Log: {log}" if log.exists() else ""), file=sys.stderr)
        sys.exit(1)
    states = ", ".join(f"{n} {c['state']}" for n, c in snap["components"].items())
    print(f"Supervisor up ({mode}, pid {snap['supervisor']['pid']}): {states}")
    if any(c["state"] != "ready" for c in snap["components"].values()):
        sys.exit(1)


def cmd_up(args) -> None:
    snap = _running()
    if snap:
        print(f"Supervisor already running ({snap['supervisor'].get('launcher', 'unknown')}, pid {snap['supervisor']['pid']}).")
        return
    use_systemd = args.systemd or (not args.manual and _unit_installed() and shutil.which("systemctl") is not None)
    if use_systemd:
        if not (_unit_installed() and shutil.which("systemctl")):
            print(f"No {UNIT} user unit installed. Run `fts service install`, or use `fts up --manual`.", file=sys.stderr)
            sys.exit(1)
        rc = _systemctl("start", UNIT)
        if rc != 0:
            print(f"systemctl --user start {UNIT} failed (exit {rc}); see `journalctl --user -u {UNIT} -n 50`.", file=sys.stderr)
            sys.exit(rc)
        _report_started("systemd")
        return
    if _unit_active():
        print(f"{UNIT}.service is active but its supervisor does not answer; use `fts doctor` or `systemctl --user restart {UNIT}`.", file=sys.stderr)
        sys.exit(1)
    cwd = Path.cwd() if (Path.cwd() / "pyproject.toml").is_file() else (repo_root() or Path.cwd())
    run_dir().mkdir(parents=True, exist_ok=True)
    log_path = run_dir() / "supervisor.log"
    with log_path.open("ab") as log:
        subprocess.Popen(  # detached: survives this terminal
            [sys.executable, "-m", "finetune_studio.supervisor", "--host", args.host, "--port", str(args.port)],
            cwd=cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    _report_started(f"manual, log {log_path}")


def cmd_down(args) -> None:
    snap = _running()
    if snap is None:
        if _unit_active():
            sys.exit(_systemctl("stop", UNIT))
        print("Supervisor is not running.")
        return
    if snap["supervisor"].get("launcher") == "systemd" and shutil.which("systemctl"):
        rc = _systemctl("stop", UNIT)
        if rc != 0:
            print(f"systemctl --user stop {UNIT} failed (exit {rc}).", file=sys.stderr)
            sys.exit(rc)
    else:
        SupervisorClient(timeout=5.0).shutdown()
    end = time.monotonic() + STOP_WAIT_S
    while time.monotonic() < end:
        if _running() is None and not socket_path().exists():
            print("Supervisor stopped.")
            return
        time.sleep(0.3)
    print("Supervisor did not stop within the wait; check `fts doctor`.", file=sys.stderr)
    sys.exit(1)


def cmd_service(args) -> None:
    root = repo_root()
    if root is None:
        print("The systemd installer ships with the repository checkout (install-service.sh); this install has none.", file=sys.stderr)
        sys.exit(1)
    flags = {"install": [], "uninstall": ["--uninstall"], "status": ["--status"], "restart": ["--restart"]}[args.action]
    if args.action == "install" and args.no_start:
        flags.append("--no-start")
    sys.exit(subprocess.run(["bash", str(root / "install-service.sh"), *flags], cwd=root, check=False, env=os.environ).returncode)
