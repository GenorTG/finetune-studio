"""Per-commit build versioning (Genor's rule 2026-09-20).

VERSION = MAJOR.MINOR.PATCH.BUILD at the repo root; the pre-commit hook bumps
BUILD on every commit, so the UI chip and /api/system/version pin the exact
deployed code. The hook MUST be pre-commit: git snapshots the commit tree
before commit-msg runs, so a bump staged there lands one commit late (the
original bug caught by the E2E). These tests lock the COMMITTED value, not
just the working file.
"""
from __future__ import annotations

import os
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


def _git(repo: Path, *args: str, env: dict | None = None) -> str:
    p = subprocess.run(["git", *args], cwd=repo, check=True,
                       capture_output=True, text=True, env=env)
    return p.stdout.strip()


def test_precommit_hook_bumps_the_COMMITTED_version(tmp_path: Path) -> None:
    """The version inside the commit tree must equal the post-hook value.

    Regression: the commit-msg variant staged 0.1.0.N+1 but committed N —
    every deployment ran one build behind its own VERSION file.
    """
    repo = tmp_path / "r"
    (repo / "scripts" / "git-hooks").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts" / "git-hooks" / "pre-commit",
                repo / "scripts" / "git-hooks" / "pre-commit")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "core.hooksPath", "scripts/git-hooks")

    def committed_version() -> str:
        return _git(repo, "show", "HEAD:VERSION")

    (repo / "a.txt").write_text("1", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "first")
    assert committed_version().endswith(".1"), committed_version()
    assert (repo / "VERSION").read_text(encoding="utf-8").strip() == committed_version()

    (repo / "a.txt").write_text("2", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "second")
    assert int(committed_version().rsplit(".", 1)[1]) == 2, committed_version()

    # env escape hatch
    before = committed_version()
    (repo / "a.txt").write_text("3", encoding="utf-8")
    _git(repo, "add", "a.txt")
    e = dict(os.environ, FTS_NO_BUMP="1")
    _git(repo, "commit", "-q", "-m", "docs only", env=e)
    assert committed_version() == before

    # a VERSION-only commit must not bump again (loop guard)
    (repo / "VERSION").write_text("9.9.9.9\n", encoding="utf-8")
    _git(repo, "add", "VERSION")
    _git(repo, "commit", "-q", "-m", "reset version")
    assert committed_version() == "9.9.9.9"


def test_hook_scripts_are_executable_and_valid_bash() -> None:
    hook = REPO_ROOT / "scripts" / "git-hooks" / "pre-commit"
    installer = REPO_ROOT / "scripts" / "install-hooks.sh"
    assert hook.stat().st_mode & 0o111, "pre-commit must be executable"
    assert installer.stat().st_mode & 0o111, "install-hooks.sh must be executable"
    for script in (hook, installer):
        r = subprocess.run(["bash", "-n", str(script)],
                           capture_output=True, text=True, check=False)
        assert r.returncode == 0, r.stderr
