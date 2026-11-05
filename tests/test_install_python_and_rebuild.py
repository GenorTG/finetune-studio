"""The installer builds on a supported Python, and a rebuild never leaves a broken environment behind.

Found by running Settings → Repair for real (2026-11-05, RESULTS §37): install.sh took Linuxbrew's Python 3.15 from PATH,
torch there could not load the system libcuda ("Found no NVIDIA driver"), llama.cpp came back as a CPU build, Heretic's
overlay stopped working, and Repair said "Finished".
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_INSTALL = (_ROOT / "install.sh").read_text(encoding="utf-8")
_UPDATE = (_ROOT / "update.sh").read_text(encoding="utf-8")


def _rebuild_function() -> str:
    m = re.search(r"^rebuild_venv\(\) \{.*?^\}", _UPDATE, re.S | re.M)
    assert m, "rebuild_venv missing from update.sh"
    return m.group(0)


def _run_rebuild(tmp_path: Path, *, install: str, healthy: bool) -> subprocess.CompletedProcess:
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "OLD").write_text("old", encoding="utf-8")
    (tmp_path / "install.sh").write_text(install, encoding="utf-8")
    script = f"""set -euo pipefail
VENV_DIR=.venv
log() {{ echo "$*"; }}; warn() {{ echo "$*"; }}; die() {{ echo "DIED: $*"; exit 1; }}
run_diagnose() {{ {'true' if healthy else 'echo the GPU is gone; false'}; }}
{_rebuild_function()}
rebuild_venv
echo COMPLETED
"""
    return subprocess.run(["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False)


_BUILDS = 'mkdir -p .venv/bin; printf "#!/bin/sh\\n" > .venv/bin/python; chmod +x .venv/bin/python; touch .venv/NEW\n'


def test_a_healthy_rebuild_replaces_the_old_environment(tmp_path: Path) -> None:
    r = _run_rebuild(tmp_path, install=_BUILDS, healthy=True)
    assert "COMPLETED" in r.stdout, r.stdout + r.stderr
    assert (tmp_path / ".venv" / "NEW").exists() and not (tmp_path / ".venv" / "OLD").exists()
    assert not (tmp_path / ".venv.before-rebuild").exists()


def test_an_unhealthy_rebuild_puts_the_old_environment_back(tmp_path: Path) -> None:
    r = _run_rebuild(tmp_path, install=_BUILDS, healthy=False)
    assert r.returncode == 1 and "COMPLETED" not in r.stdout
    assert "previous environment was put back" in r.stdout
    assert (tmp_path / ".venv" / "OLD").exists() and not (tmp_path / ".venv" / "NEW").exists()


def test_a_failed_install_puts_the_old_environment_back(tmp_path: Path) -> None:
    r = _run_rebuild(tmp_path, install="exit 7\n", healthy=True)
    assert r.returncode == 1 and "previous environment was put back" in r.stdout
    assert (tmp_path / ".venv" / "OLD").exists()


def test_install_builds_on_uvs_python_and_only_accepts_supported_ones() -> None:
    assert '--python-preference only-managed' in _INSTALL
    assert 'case "$v" in 3.12|3.13)' in _INSTALL
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12,<3.14"' in pyproject   # the two must move together


def test_repair_no_longer_deletes_first() -> None:
    assert 'rm -rf "$VENV_DIR"\n    bash install.sh' not in _UPDATE
    assert "rebuild_venv\nfi" in _UPDATE
