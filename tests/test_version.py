"""Per-commit build versioning (Genor's rule 2026-09-20).

VERSION = MAJOR.MINOR.PATCH.BUILD at the repo root; the pre-commit hook bumps
BUILD on every commit, so the UI chip and /api/system/version pin the exact
deployed code. These tests lock the format, the API, and the hook itself.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_version_format_and_file_agreement() -> None:
    from finetune_studio import __version__, build_version

    assert _VERSION_RE.match(__version__), __version__
    assert __version__ == build_version()
    assert __version__ == (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()


def test_system_version_endpoint(client) -> None:
    from finetune_studio import __version__

    r = client.get("/api/system/version")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == __version__
    assert _VERSION_RE.match(body["version"])
    assert body["channel"]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True,
                   capture_output=True, text=True)


def test_commitmsg_hook_bumps_every_commit(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    (repo / "scripts" / "git-hooks").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts" / "git-hooks" / "commit-msg",
                repo / "scripts" / "git-hooks" / "commit-msg")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "core.hooksPath", "scripts/git-hooks")

    def version_file() -> str:
        return (repo / "VERSION").read_text(encoding="utf-8").strip()

    (repo / "a.txt").write_text("1", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "first")
    assert version_file().endswith(".1"), version_file()

    (repo / "a.txt").write_text("2", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "second")
    assert int(version_file().rsplit(".", 1)[1]) == 2, version_file()

    # [no-bump] escape hatch
    before = version_file()
    (repo / "a.txt").write_text("3", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "docs only [no-bump]")
    assert version_file() == before

    # a VERSION-only commit must not bump (loop guard)
    (repo / "VERSION").write_text("9.9.9.9\n", encoding="utf-8")
    _git(repo, "add", "VERSION")
    _git(repo, "commit", "-q", "-m", "reset version")
    assert version_file() == "9.9.9.9"


def test_hook_scripts_are_executable_and_valid_bash() -> None:
    hook = REPO_ROOT / "scripts" / "git-hooks" / "commit-msg"
    installer = REPO_ROOT / "scripts" / "install-hooks.sh"
    assert hook.stat().st_mode & 0o111, "commit-msg must be executable"
    assert installer.stat().st_mode & 0o111, "install-hooks.sh must be executable"
    for script in (hook, installer):
        r = subprocess.run(["bash", "-n", str(script)],
                           capture_output=True, text=True, check=False)
        assert r.returncode == 0, r.stderr
