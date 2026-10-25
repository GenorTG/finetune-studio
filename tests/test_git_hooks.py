"""scripts/git-hooks: pre-commit + pre-push enforce the branch rules locally (CONTRIBUTING.md "Branches")."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOKS = REPO_ROOT / "scripts" / "git-hooks"
ZERO = "0" * 40


def git(repo: Path, *args: str, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, check=check, capture_output=True, text=True,
                          env={**os.environ, **(env or {})})


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    (r / "scripts" / "git-hooks").mkdir(parents=True)
    for f in ("pre-commit", "pre-push"):
        shutil.copy(HOOKS / f, r / "scripts" / "git-hooks" / f)
    shutil.copy(REPO_ROOT / "scripts" / "branch_policy.py", r / "scripts" / "branch_policy.py")
    git(r, "init", "-q", "-b", "dev")
    git(r, "config", "user.email", "t@t")
    git(r, "config", "user.name", "t")
    (r / "VERSION").write_text("1.0.0\n")
    git(r, "add", ".")
    git(r, "commit", "-q", "-m", "init")  # before the hooks are on
    git(r, "config", "core.hooksPath", "scripts/git-hooks")
    return r


def _commit(repo: Path, name: str = "a.txt", env: dict | None = None) -> subprocess.CompletedProcess:
    (repo / name).write_text(name + str(len(list(repo.iterdir()))))
    git(repo, "add", name)
    return git(repo, "commit", "-q", "-m", "c", env=env, check=False)


def test_hooks_are_executable_bash() -> None:
    for f in ("pre-commit", "pre-push"):
        p = HOOKS / f
        assert os.access(p, os.X_OK), f
        assert subprocess.run(["bash", "-n", str(p)], check=False).returncode == 0, f


@pytest.mark.parametrize("branch", ["dev", "main"])
def test_no_commit_on_protected_branches(repo: Path, branch: str) -> None:
    git(repo, "switch", "-q", "-C", branch)
    p = _commit(repo)
    assert p.returncode != 0 and f"no commits on '{branch}'" in p.stderr
    assert _commit(repo, "b.txt", env={"FTS_ALLOW_PROTECTED_COMMIT": "1"}).returncode == 0


def test_commit_on_a_work_branch(repo: Path) -> None:
    git(repo, "switch", "-q", "-c", "feat/x")
    assert _commit(repo).returncode == 0


@pytest.mark.parametrize("branch,ok", [("feat/x", False), ("release/v1.1.0", True), ("hotfix/v1.0.1", True)])
def test_version_changes_only_on_release_branches(repo: Path, branch: str, ok: bool) -> None:
    git(repo, "switch", "-q", "-c", branch)
    (repo / "VERSION").write_text("1.1.0\n")
    git(repo, "add", "VERSION")
    p = git(repo, "commit", "-q", "-m", "bump", check=False)
    assert (p.returncode == 0) is ok, p.stderr


def _pre_push(repo: Path, remote_ref: str, local_sha: str | None = None,
              env: dict | None = None) -> subprocess.CompletedProcess:
    sha = local_sha or git(repo, "rev-parse", "HEAD").stdout.strip()
    line = f"refs/heads/x {sha} {remote_ref} {ZERO}\n"
    return subprocess.run(["bash", str(repo / "scripts" / "git-hooks" / "pre-push"), "origin", "url"],
                          cwd=repo, input=line, capture_output=True, text=True, check=False,
                          env={**os.environ, **(env or {})})


@pytest.mark.parametrize("ref,ok", [
    ("refs/heads/feat/use-case-ui", True), ("refs/heads/release/v1.1.0", True), ("refs/heads/sync/v1.1.0", True),
    ("refs/heads/dev", False), ("refs/heads/main", False),
    ("refs/heads/full-history", False), ("refs/heads/backup/pre-history-rewrite", False),
    ("refs/heads/my-branch", False), ("refs/heads/feature/x", False),
    ("refs/tags/v1.1.0", True), ("refs/tags/v1.1.0-rc.1", True), ("refs/tags/latest", False),
])
def test_pre_push_rules(repo: Path, ref: str, ok: bool) -> None:
    p = _pre_push(repo, ref)
    assert (p.returncode == 0) is ok, p.stderr


def test_pre_push_escape_covers_main_dev_but_never_the_archive(repo: Path) -> None:
    env = {"FTS_ALLOW_PROTECTED_PUSH": "1"}
    assert _pre_push(repo, "refs/heads/dev", env=env).returncode == 0
    assert _pre_push(repo, "refs/heads/full-history", env=env).returncode != 0


def test_deleting_a_remote_branch_is_allowed(repo: Path) -> None:
    assert _pre_push(repo, "refs/heads/old-name", local_sha=ZERO).returncode == 0
