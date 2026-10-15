"""install-service.sh, run for real against stub systemctl / loginctl / sudo.

2026-10-15: ``--uninstall`` ended with ``pkill -f "uvicorn finetune_studio.webui.app:app"`` and killed every other
instance on the box (the live web child of another checkout); ``--system`` wrote /etc without sudo, ran its
systemctl calls without sudo and left the unit running as root with ``%h`` = /root.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "install-service.sh"
STUB = """#!/bin/sh
echo "{name} $*" >> "{log}"
if [ "{name}" = sudo ]; then
  if [ "$1" = tee ]; then cat > "{root}/system.unit"; exit 0; fi
  if [ "$1" = mkdir ]; then exit 0; fi
  exec "$@"
fi
exit 0
"""


@pytest.fixture
def box(tmp_path: Path) -> Path:
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    shutil.copy(SCRIPT, tmp_path / "install-service.sh")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    py = tmp_path / ".venv" / "bin" / "python"          # a wrapper: a symlinked venv python loses its site-packages
    py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    (tmp_path / "bin").mkdir()
    for name in ("systemctl", "loginctl", "sudo", "pkill", "curl"):
        stub = tmp_path / "bin" / name
        stub.write_text(STUB.format(name=name, log=tmp_path / "calls.log", root=tmp_path))
        stub.chmod(0o755)
    (tmp_path / "home").mkdir()
    return tmp_path


def _run(box: Path, *args: str) -> tuple[int, list[str]]:
    env = {**os.environ, "PATH": f"{box / 'bin'}:{os.environ['PATH']}", "HOME": str(box / "home"),
           "XDG_CONFIG_HOME": str(box / "home" / ".config")}
    env.pop("FTS_ROOT", None)
    r = subprocess.run(["bash", str(box / "install-service.sh"), *args], env=env, capture_output=True, text=True,
                       timeout=60)
    log = box / "calls.log"
    return r.returncode, (log.read_text().splitlines() if log.exists() else [])


def test_user_install_writes_a_portable_unit_and_uninstall_kills_nothing_else(box: Path) -> None:
    rc, calls = _run(box, "--no-start")
    unit = (box / "home" / ".config" / "systemd" / "user" / "finetune-studio.service").read_text()
    assert rc == 0 and "systemctl --user enable finetune-studio" in calls
    assert "Environment=FTS_ROOT=%h/.finetune-studio" in unit and "User=" not in unit
    rc, calls = _run(box, "--uninstall")
    assert rc == 0 and "systemctl --user disable --now finetune-studio" in calls
    assert not any(c.startswith("pkill") for c in calls)


def test_system_install_goes_through_sudo_and_runs_as_the_installing_user(box: Path) -> None:
    rc, calls = _run(box, "--system", "--no-start")
    assert rc == 0, calls
    assert "sudo tee /etc/systemd/system/finetune-studio.service" in calls
    assert "sudo systemctl daemon-reload" in calls and "sudo systemctl enable finetune-studio" in calls
    unit = (box / "system.unit").read_text()
    user = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
    assert f"User={user}" in unit
    assert f"Environment=FTS_ROOT={box / 'home'}/.finetune-studio" in unit and "%h" not in unit
