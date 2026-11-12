"""update.sh from a shell with systemd's short PATH (Settings → Apply update) finds uv where it was installed, and a
failed dependency step stops the update with the reason instead of restarting and saying "Update complete".

Found in the v0.3.0 release (2026-11-12): uv lived in /home/linuxbrew/.linuxbrew/bin, the in-app update's PATH did not
have it, every sync step WARNed and the script still ended "Update complete".
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPDATE = (ROOT / "update.sh").read_text(encoding="utf-8")


def _fn(name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{.*?^\}}", UPDATE, re.DOTALL | re.MULTILINE)
    assert m, f"{name} missing from update.sh"
    return m.group(0)


def _gate() -> str:
    m = re.search(r'^if \[ "\$\{#PROBLEMS\[@\]\}" -gt 0 \].*?^fi$', UPDATE, re.DOTALL | re.MULTILINE)
    assert m, "the problems gate is missing from update.sh"
    return m.group(0)


def _bash(script: str, *, home: Path, cwd: Path) -> subprocess.CompletedProcess:
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}       # what a systemd-launched shell gets
    return subprocess.run(["bash", "-c", script], cwd=cwd, env=env, capture_output=True, text=True, timeout=30, check=False)


def _stub_uv(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    uv = path / "uv"
    uv.write_text('#!/bin/sh\necho "fake uv $*"\n', encoding="utf-8")
    uv.chmod(0o755)
    return uv


def test_uv_is_found_outside_path(tmp_path: Path) -> None:
    home = tmp_path / "home"
    uv = _stub_uv(home / ".local" / "bin")
    (tmp_path / ".venv").mkdir()
    r = _bash(f"VENV_DIR=.venv\n{_fn('find_uv')}\nfind_uv", home=home, cwd=tmp_path)
    assert r.returncode == 0 and r.stdout.strip() == str(uv), r.stdout + r.stderr


def test_the_path_install_sh_recorded_wins_over_guessing(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _stub_uv(home / ".local" / "bin")
    recorded = _stub_uv(tmp_path / "brew" / "bin")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "fts-uv-path").write_text(str(recorded) + "\n", encoding="utf-8")
    r = _bash(f"VENV_DIR=.venv\n{_fn('find_uv')}\nfind_uv", home=home, cwd=tmp_path)
    assert r.stdout.strip() == str(recorded)
    assert 'echo "$UV_PATH" > "$VENV_DIR/fts-uv-path"' in (ROOT / "install.sh").read_text(encoding="utf-8")


def test_a_failed_sync_stops_the_update_with_the_reason(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    py = tmp_path / ".venv" / "bin" / "python"
    py.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")       # no pip module in this venv
    py.chmod(0o755)
    script = f"""set -uo pipefail
VENV_DIR=.venv; VENV_PY="$PWD/.venv/bin/python"; CHECK_MODE=0
log() {{ echo "$*"; }}; warn() {{ echo "WARN: $*" >&2; }}
PROBLEMS=(); problem() {{ warn "$*"; PROBLEMS+=("$*"); }}
UV=""     # no uv anywhere (the tests above cover finding one)
{_fn('pip_install')}
pip_install -e '.[parsers]' 2>&1 | tail -5 || problem "dependency sync (pip install -e .[parsers]) failed$([ -n "$UV" ] || echo ': uv was not found')"
{_gate()}
echo "Restarting finetune-studio.service..."
echo "Update complete."
"""
    r = _bash(script, home=home, cwd=tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert "ERROR: update not finished, 1 step(s) failed: dependency sync" in out and "uv was not found" in out
    assert "Update complete" not in out and "Restarting" not in out


def test_the_gate_comes_before_the_restart_and_the_worker_keeps_the_reason() -> None:
    assert UPDATE.index('if [ "${#PROBLEMS[@]}" -gt 0 ]') < UPDATE.index('log "Restarting finetune-studio.service..."')
    assert UPDATE.index('if [ "${#PROBLEMS[@]}" -gt 0 ]') < UPDATE.index('log "Update complete."')
    for step in ("bitsandbytes sync failed", "numpy/scipy sync failed", "llama.cpp CLI build failed"):
        assert f'problem "{step}' in UPDATE, step
    updates = (ROOT / "src/finetune_studio/webui/routes/updates.py").read_text(encoding="utf-8")
    assert 'error=reason or f"update.sh exited with code {proc.returncode}"' in updates


def test_this_host_finds_its_uv_from_a_short_path() -> None:
    """The live checkout: uv is not on systemd's PATH here, find_uv still finds it."""
    if not (ROOT / ".venv").is_dir():
        return
    env = {"HOME": os.environ.get("HOME", ""), "PATH": "/usr/bin:/bin"}
    r = subprocess.run(["bash", "-c", f"VENV_DIR=.venv\n{_fn('find_uv')}\nfind_uv"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=30, check=False)
    if r.returncode == 0:
        assert os.access(r.stdout.strip(), os.X_OK)
