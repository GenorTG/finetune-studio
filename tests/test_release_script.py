"""scripts/release.py: prepare -> (merge) -> finish -> verify-tag, against a local bare "origin" (no gh)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "release.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import release as rel  # noqa: E402

CHANGELOG = """# Changelog

## [Unreleased]

### Added
- Use-case UI.

## [0.2.0] - 2026-10-25

### Added
- First release.

[Unreleased]: https://github.com/GenorTG/finetune-studio/compare/v0.2.0...dev
[0.2.0]: https://github.com/GenorTG/finetune-studio/releases/tag/v0.2.0
"""
PYPROJECT = '[project]\nname = "finetune-studio"\nversion = "0.2.0"\n'


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def run(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True,
                          check=False)


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    work = tmp_path / "work"
    git(tmp_path, "clone", "-q", str(origin), str(work))
    for k, v in (("user.email", "t@t"), ("user.name", "t"), ("core.hooksPath", "/dev/null")):
        git(work, "config", k, v)
    (work / "VERSION").write_text("0.2.0\n")
    (work / "pyproject.toml").write_text(PYPROJECT)
    (work / "CHANGELOG.md").write_text(CHANGELOG)
    git(work, "add", ".")
    git(work, "commit", "-q", "-m", "v0.2.0")
    git(work, "branch", "-M", "main")
    git(work, "tag", "-a", "v0.2.0", "-m", "rel")
    git(work, "push", "-q", "origin", "main", "v0.2.0")
    git(work, "switch", "-q", "-c", "dev")
    (work / "feature.py").write_text("x = 1\n")
    git(work, "add", "feature.py")
    git(work, "commit", "-q", "-m", "Use-case UI")
    git(work, "push", "-q", "-u", "origin", "dev")
    return work


@pytest.mark.parametrize("cur,how,new", [
    ("0.2.0", "minor", "0.3.0"), ("0.2.0", "patch", "0.2.1"), ("0.2.0", "major", "1.0.0"),
    ("1.0.0-rc.2", "patch", "1.0.1"), ("0.2.0", "1.0.0-rc.1", "1.0.0-rc.1"),
])
def test_bump(cur: str, how: str, new: str) -> None:
    assert rel.bump(cur, how) == new


def test_bump_rejects_nonsense() -> None:
    with pytest.raises(rel.ReleaseError):
        rel.bump("0.2.0", "huge")


def test_changelog_section() -> None:
    assert rel.changelog_section(CHANGELOG, "Unreleased") == "### Added\n- Use-case UI."
    assert rel.changelog_section(CHANGELOG, "0.2.0") == "### Added\n- First release."
    assert rel.changelog_section(CHANGELOG, "9.9.9") == ""


def test_date_changelog_refuses_an_empty_release(clone: Path) -> None:
    empty = "# Changelog\n\n## [Unreleased]\n\n## [0.2.0] - 2026-10-25\n- x\n"
    with pytest.raises(rel.ReleaseError, match="empty"):
        rel.date_changelog(empty, "0.3.0", "0.2.0", "2026-11-01")


def test_full_release_cycle(clone: Path, tmp_path: Path) -> None:
    origin = tmp_path / "origin.git"
    p = run(clone, "prepare", "minor", "--no-pr", "--date", "2026-11-01")
    assert p.returncode == 0, p.stderr
    assert git(clone, "rev-parse", "--abbrev-ref", "HEAD") == "release/v0.3.0"
    assert (clone / "VERSION").read_text().strip() == "0.3.0"
    assert 'version = "0.3.0"' in (clone / "pyproject.toml").read_text()
    cl = (clone / "CHANGELOG.md").read_text()
    assert "## [Unreleased]\n\n## [0.3.0] - 2026-11-01\n\n### Added\n- Use-case UI." in cl
    assert "[Unreleased]: https://github.com/GenorTG/finetune-studio/compare/v0.3.0...dev" in cl
    assert "[0.3.0]: https://github.com/GenorTG/finetune-studio/compare/v0.2.0...v0.3.0" in cl
    assert "[0.2.0]: https://github.com/GenorTG/finetune-studio/releases/tag/v0.2.0" in cl
    assert git(origin, "rev-parse", "refs/heads/release/v0.3.0")  # pushed
    assert git(clone, "log", "-1", "--format=%s") == "release: v0.3.0"

    # finishing before the merge is refused
    p = run(clone, "finish", "0.3.0", "--no-pr")
    assert p.returncode == 1 and "merge the release PR first" in p.stderr

    # what GitHub does on "merge commit"
    git(clone, "switch", "-q", "main")
    git(clone, "merge", "-q", "--no-ff", "-m", "Merge release/v0.3.0", "release/v0.3.0")
    git(clone, "push", "-q", "origin", "main")
    # dev moved on meanwhile
    git(clone, "switch", "-q", "dev")
    (clone / "later.py").write_text("y = 2\n")
    git(clone, "add", "later.py")
    git(clone, "commit", "-q", "-m", "later work")
    git(clone, "push", "-q", "origin", "dev")

    p = run(clone, "finish", "0.3.0", "--no-pr")
    assert p.returncode == 0, p.stderr
    assert git(origin, "rev-parse", "v0.3.0^{commit}") == git(origin, "rev-parse", "main")
    assert git(origin, "cat-file", "-t", "v0.3.0") == "tag"  # annotated
    assert git(clone, "rev-parse", "--abbrev-ref", "HEAD") == "sync/v0.3.0"
    assert subprocess.run(["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], cwd=clone,
                          check=False).returncode == 0
    assert (clone / "later.py").exists() and (clone / "VERSION").read_text().strip() == "0.3.0"
    assert git(origin, "rev-parse", "refs/heads/sync/v0.3.0")

    p = run(clone, "verify-tag", "v0.3.0", "--main-ref", "origin/main")
    assert p.returncode == 0, p.stdout
    p = run(clone, "notes", "0.3.0")
    assert p.stdout.strip() == "### Added\n- Use-case UI."


def test_prepare_refuses_a_dirty_tree(clone: Path) -> None:
    (clone / "feature.py").write_text("x = 2\n")
    p = run(clone, "prepare", "minor", "--no-pr")
    assert p.returncode == 1 and "uncommitted" in p.stderr


def test_verify_tag_catches_a_mismatch(clone: Path) -> None:
    git(clone, "switch", "-q", "main")
    git(clone, "tag", "-a", "v0.9.0", "-m", "wrong")
    p = run(clone, "verify-tag", "v0.9.0", "--main-ref", "main")
    assert p.returncode == 1
    assert "VERSION at v0.9.0 is '0.2.0'" in p.stdout and "no [0.9.0] section" in p.stdout


def test_hotfix_bumps_in_place(clone: Path) -> None:
    git(clone, "switch", "-q", "-c", "hotfix/v0.2.1", "origin/main")
    text = (clone / "CHANGELOG.md").read_text().replace(
        "## [Unreleased]\n\n### Added\n- Use-case UI.\n", "## [Unreleased]\n\n### Fixed\n- Crash.\n")
    (clone / "CHANGELOG.md").write_text(text)
    git(clone, "commit", "-q", "-am", "Fix crash")
    p = run(clone, "prepare", "patch", "--no-pr", "--date", "2026-11-02")
    assert p.returncode == 0, p.stderr
    assert git(clone, "rev-parse", "--abbrev-ref", "HEAD") == "hotfix/v0.2.1"
    assert (clone / "VERSION").read_text().strip() == "0.2.1"
    assert "## [0.2.1] - 2026-11-02\n\n### Fixed\n- Crash." in (clone / "CHANGELOG.md").read_text()
