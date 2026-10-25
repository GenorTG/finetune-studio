"""Branch rules for PRs (scripts/branch_policy.py, the required `branch-policy` check; CONTRIBUTING.md)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import branch_policy as bp  # noqa: E402

CL_RELEASE = "## [Unreleased]\n\n## [1.3.0] - 2026-11-02\n\n### Added\n- x\n"


def pr(base: str, head: str, changed=("src/a.py",), head_version="1.2.0", base_version="1.2.0",
       changelog="", labels=()) -> bp.PullRequest:
    return bp.PullRequest(base=base, head=head, changed=list(changed), head_version=head_version,
                          base_version=base_version, head_changelog=changelog, labels=list(labels))


@pytest.mark.parametrize("name,ok", [
    ("feat/use-case-ui", True), ("fix/mining.direction_2", True), ("chore/x", True), ("docs/y", True),
    ("refactor/z", True), ("test/a", True), ("ci/b", True), ("perf/c", True), ("build/d", True),
    ("release/v1.2.0", True), ("hotfix/v1.2.1", True), ("sync/v1.2.0", True), ("release/v1.0.0-rc.1", True),
    ("dependabot/github_actions/actions-abc", True),
    ("feature/x", False), ("feat/", False), ("feat/Upper", False), ("feat/a/b", False), ("wip", False),
    ("main2", False), ("release/1.2.0", False), ("release/v1.2", False), ("full-history", False),
    ("backup/pre-history-rewrite", False),
])
def test_branch_names(name: str, ok: bool) -> None:
    assert bp.branch_name_ok(name) is ok


def test_semver_order() -> None:
    keys = ["0.1.0", "0.2.0", "1.0.0-beta.2", "1.0.0-beta.10", "1.0.0-rc.1", "1.0.0", "1.0.1", "1.10.0"]
    assert sorted(keys, key=bp.parse_semver) == keys
    with pytest.raises(ValueError):
        bp.parse_semver("0.1.0.532")


def test_work_branch_into_dev() -> None:
    assert bp.check(pr("dev", "chore/x")) == []
    assert bp.check(pr("dev", "feat/x", changed=["src/a.py", "CHANGELOG.md"])) == []


def test_feat_fix_perf_need_a_changelog_line_or_the_label() -> None:
    for head in ("feat/x", "fix/x", "perf/x"):
        errs = bp.check(pr("dev", head))
        assert any("CHANGELOG" in e for e in errs), head
        assert bp.check(pr("dev", head, labels=["skip-changelog"])) == []


def test_work_pr_never_changes_version() -> None:
    errs = bp.check(pr("dev", "chore/x", changed=["VERSION"], head_version="1.3.0"))
    assert any("VERSION" in e for e in errs)


def test_one_time_migration_off_the_build_counter() -> None:
    assert bp.check(pr("dev", "chore/release-system", changed=["VERSION"], head_version="0.1.0",
                       base_version="0.1.0.532")) == []
    # once dev is SemVer the exception is gone
    assert bp.check(pr("dev", "chore/y", changed=["VERSION"], head_version="0.1.1", base_version="0.1.0"))


@pytest.mark.parametrize("head", ["main", "dev"])
def test_protected_branch_as_head_is_refused(head: str) -> None:
    assert bp.check(pr("main" if head == "dev" else "dev", head))


def test_bad_names_into_dev() -> None:
    assert bp.check(pr("dev", "my-branch"))
    assert any("merges into main" in e for e in bp.check(pr("dev", "release/v1.3.0")))
    assert bp.check(pr("dev", "sync/v1.3.0", changed=["VERSION"], head_version="1.3.0")) == []
    assert bp.check(pr("dev", "dependabot/github_actions/x")) == []


def test_only_release_or_hotfix_into_main() -> None:
    assert any("only release/vX.Y.Z" in e for e in bp.check(pr("main", "feat/x")))
    assert bp.check(pr("main", "sync/v1.3.0"))


def test_release_into_main() -> None:
    ok = pr("main", "release/v1.3.0", changed=["VERSION", "CHANGELOG.md"], head_version="1.3.0",
            base_version="1.2.0", changelog=CL_RELEASE)
    assert bp.check(ok) == []
    assert bp.check(pr("main", "hotfix/v1.3.0", head_version="1.3.0", base_version="1.2.0",
                       changelog=CL_RELEASE)) == []


def test_release_into_main_rules() -> None:
    errs = bp.check(pr("main", "release/v1.3.0", head_version="1.2.9", base_version="1.2.0",
                       changelog=CL_RELEASE))
    assert any("branch name says" in e for e in errs)
    errs = bp.check(pr("main", "release/v1.3.0", head_version="1.3.0", base_version="1.3.0",
                       changelog=CL_RELEASE))
    assert any("not newer" in e for e in errs)
    errs = bp.check(pr("main", "release/v1.3.0", head_version="1.3.0", base_version="1.2.0",
                       changelog="## [Unreleased]\n- x\n"))
    assert any("CHANGELOG.md has no" in e for e in errs)


def test_first_release_replaces_the_legacy_version_on_main() -> None:
    assert bp.check(pr("main", "release/v0.2.0", head_version="0.2.0", base_version="0.1.0.532",
                       changelog="## [0.2.0] - 2026-10-25\n- x\n")) == []


def test_other_bases_are_unrestricted() -> None:
    assert bp.check(pr("feat/base", "whatever")) == []


def test_cli_against_a_real_repo(tmp_path: Path) -> None:
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True, text=True).stdout

    git("init", "-q", "-b", "dev")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    git("config", "core.hooksPath", "/dev/null")
    (tmp_path / "VERSION").write_text("1.2.0\n")
    git("add", "VERSION")
    git("commit", "-q", "-m", "base")
    git("switch", "-q", "-c", "fix/a")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", "a.py")
    git("commit", "-q", "-m", "fix")
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "branch_policy.py"), "--base", "dev", "--head", "fix/a",
           "--base-ref", "dev"]
    p = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert p.returncode == 1 and "CHANGELOG" in p.stdout
    p = subprocess.run([*cmd, "--labels", "skip-changelog"], cwd=tmp_path, capture_output=True, text=True,
                       check=False)
    assert p.returncode == 0, p.stdout
