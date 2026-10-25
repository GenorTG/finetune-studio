"""SemVer release version + build identity (CONTRIBUTING.md "Versions and releases").

VERSION holds MAJOR.MINOR.PATCH[-pre] and only scripts/release.py changes it. The app shows the plain
version on the tagged release commit and ``<version>+<branch>.<sha8>`` everywhere else, read from
``.git`` files without a subprocess (loose refs, packed refs, annotated tags, linked worktrees).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from finetune_studio import version as V

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-q", "-b", "dev")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "core.hooksPath", "/dev/null")
    (r / "VERSION").write_text("1.4.0\n")
    _git(r, "add", "VERSION")
    _git(r, "commit", "-q", "-m", "init")
    return r


def test_repo_version_file_is_semver_and_matches_pyproject() -> None:
    v = (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    assert V.SEMVER_RE.match(v), v
    py = re.search(r'(?m)^version\s*=\s*"([^"]*)"', (REPO_ROOT / "pyproject.toml").read_text())
    assert py and py.group(1) == v, "pyproject version must equal VERSION (scripts/release.py sets both)"


def test_package_exports_release_and_build() -> None:
    import finetune_studio as fs

    assert fs.__release__ == V.release_version()
    assert fs.__version__.split("+", 1)[0] == fs.__release__
    assert fs.__release_channel__ in ("BETA", "STABLE")


def test_system_version_endpoint(client) -> None:
    r = client.get("/api/system/version")
    assert r.status_code == 200
    body = r.json()
    assert body["release"] == V.release_version()
    assert body["version"] == V.build_version()
    assert body["channel"] == V.release_channel(body["release"])
    assert isinstance(body["prerelease"], bool)
    assert set(body) >= {"git_commit", "git_branch"}


@pytest.mark.parametrize("v,pre,channel", [
    ("0.2.0", True, "BETA"), ("1.0.0-rc.1", True, "BETA"), ("1.0.0", False, "STABLE"),
    ("2.3.4", False, "STABLE"), ("0.1.0.532", True, "BETA"),
])
def test_prerelease_and_channel(v: str, pre: bool, channel: str) -> None:
    assert V.is_prerelease(v) is pre
    assert V.release_channel(v) == channel


def test_untagged_commit_shows_branch_and_sha(repo: Path) -> None:
    sha = _git(repo, "rev-parse", "HEAD")
    assert V.git_info(repo) == V.GitInfo(branch="dev", commit=sha, tags=())
    assert V.build_version(repo) == f"1.4.0+dev.{sha[:8]}"


def test_branch_slug_is_build_metadata_safe(repo: Path) -> None:
    _git(repo, "switch", "-q", "-c", "feat/x_y")
    sha = _git(repo, "rev-parse", "HEAD")
    assert V.build_version(repo) == f"1.4.0+feat-x-y.{sha[:8]}"


@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_tagged_release_commit_shows_the_plain_version(repo: Path, annotated: bool, packed: bool) -> None:
    if annotated:
        _git(repo, "tag", "-a", "v1.4.0", "-m", "rel")
    else:
        _git(repo, "tag", "v1.4.0")
    if packed:
        _git(repo, "pack-refs", "--all")
        assert not (repo / ".git" / "refs" / "tags" / "v1.4.0").exists()
    assert "v1.4.0" in V.git_info(repo).tags
    assert V.build_version(repo) == "1.4.0"
    (repo / "f").write_text("x")
    _git(repo, "add", "f")
    _git(repo, "commit", "-q", "-m", "after")
    assert V.build_version(repo).startswith("1.4.0+dev.")


def test_packed_branch_ref_and_detached_head(repo: Path) -> None:
    sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "pack-refs", "--all")
    assert V.git_info(repo).commit == sha
    _git(repo, "switch", "-q", "--detach")
    assert V.build_version(repo) == f"1.4.0+detached.{sha[:8]}"


def test_linked_worktree(repo: Path, tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", "-b", "fix/a", str(wt))
    sha = _git(wt, "rev-parse", "HEAD")
    assert (wt / ".git").is_file()
    assert V.git_info(wt).branch == "fix/a"
    assert V.build_version(wt) == f"1.4.0+fix-a.{sha[:8]}"


def test_no_git_metadata_reports_the_plain_version(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text("3.1.4\n")
    assert V.git_info(tmp_path) == V.GitInfo()
    assert V.build_version(tmp_path) == "3.1.4"


def test_missing_version_file_falls_back_to_package_metadata(tmp_path: Path) -> None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        expected = version("finetune-studio")
    except PackageNotFoundError:
        expected = "0.0.0"
    assert V.release_version(tmp_path) == expected
