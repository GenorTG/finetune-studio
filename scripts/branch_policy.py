#!/usr/bin/env python3
"""Branch rules for pull requests (CONTRIBUTING.md "Branches"); the required `branch-policy` check.

    main  <- release/vX.Y.Z | hotfix/vX.Y.Z        (the release branch: tagged vX.Y.Z after merge)
    dev   <- feat/* fix/* chore/* docs/* refactor/* test/* ci/* perf/* build/* | sync/vX.Y.Z | dependabot/*

Into main: VERSION on the branch must equal the branch's X.Y.Z, be newer than main's, and
CHANGELOG.md must have its `## [X.Y.Z] - YYYY-MM-DD` section. Into dev: VERSION never changes
(only release/hotfix branches carry a bump; sync/ brings it back from main), and feat/fix/perf
branches add a CHANGELOG line under `## [Unreleased]` unless the PR has the `skip-changelog` label.

Usage (CI): branch_policy.py --base main --head release/v1.2.0 --base-ref origin/main [--labels a,b]
Stdlib only: the CI job runs it with the runner's python3, no venv.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field

SEMVER = r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
WORK_PREFIXES = ("feat", "fix", "chore", "docs", "refactor", "test", "ci", "perf", "build")
WORK_RE = re.compile(rf"^(?:{'|'.join(WORK_PREFIXES)})/[a-z0-9][a-z0-9._-]*$")
RELEASE_RE = re.compile(rf"^(?P<kind>release|hotfix)/v(?P<version>{SEMVER})$")
SYNC_RE = re.compile(rf"^sync/v(?P<version>{SEMVER})$")
DEPENDABOT_RE = re.compile(r"^dependabot/")
NEEDS_CHANGELOG = ("feat/", "fix/", "perf/")
PROTECTED = ("main", "dev")
SKIP_CHANGELOG_LABEL = "skip-changelog"


def parse_semver(v: str) -> tuple:
    """Sortable key; a pre-release sorts before its release (1.0.0-rc.1 < 1.0.0)."""
    m = re.fullmatch(rf"({SEMVER})", v.strip())
    if not m:
        raise ValueError(f"not a SemVer version: {v!r}")
    core, _, pre = v.strip().partition("-")
    major, minor, patch = (int(x) for x in core.split("."))
    if not pre:
        return (major, minor, patch, 1, ())
    ids = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in pre.split("."))
    return (major, minor, patch, 0, ids)


def is_semver(v: str) -> bool:
    try:
        parse_semver(v)
        return True
    except ValueError:
        return False


def branch_name_ok(name: str) -> bool:
    """Any branch that may exist on GitHub besides main/dev."""
    return bool(WORK_RE.match(name) or RELEASE_RE.match(name) or SYNC_RE.match(name)
                or DEPENDABOT_RE.match(name))


def changelog_has_release(changelog: str, version: str) -> bool:
    return re.search(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\s*$", changelog, re.MULTILINE) is not None


@dataclass
class PullRequest:
    base: str
    head: str
    changed: list[str]
    head_version: str
    base_version: str
    head_changelog: str = ""
    labels: list[str] = field(default_factory=list)


def check(pr: PullRequest) -> list[str]:
    """Every rule the PR breaks, as one readable line each; empty = allowed."""
    errors: list[str] = []
    if pr.base not in PROTECTED:
        return errors  # stacked PRs between work branches: no rules
    if pr.head in PROTECTED:
        errors.append(f"'{pr.head}' -> '{pr.base}' is not a PR this repo takes: "
                      "release with scripts/release.py (release/vX.Y.Z -> main), "
                      "sync main into dev with sync/vX.Y.Z")
        return errors
    if pr.base == "main":
        m = RELEASE_RE.match(pr.head)
        if not m:
            errors.append(f"only release/vX.Y.Z or hotfix/vX.Y.Z branches merge into main, not '{pr.head}' "
                          "(work goes into dev; `make release-prepare BUMP=minor` cuts a release)")
            return errors
        v = m.group("version")
        if pr.head_version != v:
            errors.append(f"VERSION on the branch is '{pr.head_version}', the branch name says '{v}'")
        try:
            head_key = parse_semver(pr.head_version)
        except ValueError as e:
            errors.append(str(e))
        else:
            if is_semver(pr.base_version) and head_key <= parse_semver(pr.base_version):
                errors.append(f"VERSION '{pr.head_version}' is not newer than main's '{pr.base_version}'")
            # a non-SemVer base is the old MAJOR.MINOR.PATCH.BUILD file: the first release replaces it
        if not changelog_has_release(pr.head_changelog, v):
            errors.append(f"CHANGELOG.md has no '## [{v}] - YYYY-MM-DD' section")
        return errors
    # base == dev
    if SYNC_RE.match(pr.head) or DEPENDABOT_RE.match(pr.head):
        return errors
    if RELEASE_RE.match(pr.head):
        errors.append(f"'{pr.head}' merges into main, not dev; after the release `make release-finish` "
                      "opens sync/vX.Y.Z -> dev")
        return errors
    if not WORK_RE.match(pr.head):
        errors.append(f"branch '{pr.head}' does not follow <type>/<slug> with type one of "
                      f"{', '.join(WORK_PREFIXES)} (lowercase slug: a-z 0-9 . _ -)")
        return errors
    if ("VERSION" in pr.changed or pr.head_version != pr.base_version) and not _semver_migration(pr):
        errors.append("VERSION changes only on release/hotfix branches (scripts/release.py), never in a work PR")
    if pr.head.startswith(NEEDS_CHANGELOG) and "CHANGELOG.md" not in pr.changed \
            and SKIP_CHANGELOG_LABEL not in pr.labels:
        errors.append("feat/fix/perf PRs add a line under '## [Unreleased]' in CHANGELOG.md "
                      f"(or carry the '{SKIP_CHANGELOG_LABEL}' label)")
    return errors


def _semver_migration(pr: PullRequest) -> bool:
    """The one-time move off the old MAJOR.MINOR.PATCH.BUILD file: base not SemVer, head SemVer."""
    return not is_semver(pr.base_version) and is_semver(pr.head_version)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def _show(ref: str, path: str) -> str:
    try:
        return _git("show", f"{ref}:{path}")
    except subprocess.CalledProcessError:
        return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, help="PR base branch name (main / dev)")
    ap.add_argument("--head", required=True, help="PR head branch name")
    ap.add_argument("--base-ref", required=True, help="git ref of the base tip, e.g. origin/main")
    ap.add_argument("--head-ref", default="HEAD", help="git ref of the head tip (default HEAD)")
    ap.add_argument("--labels", default="", help="comma-separated PR labels")
    a = ap.parse_args(argv)
    changed = [p for p in _git("diff", "--name-only", f"{a.base_ref}...{a.head_ref}").splitlines() if p]
    pr = PullRequest(
        base=a.base, head=a.head, changed=changed,
        head_version=_show(a.head_ref, "VERSION").strip(),
        base_version=_show(a.base_ref, "VERSION").strip(),
        head_changelog=_show(a.head_ref, "CHANGELOG.md"),
        labels=[x.strip() for x in a.labels.split(",") if x.strip()],
    )
    errors = check(pr)
    if errors:
        for e in errors:
            print(f"::error::{e}")
        print(f"branch-policy: {a.head} -> {a.base} REFUSED ({len(errors)} rule(s)); see CONTRIBUTING.md")
        return 1
    print(f"branch-policy: {a.head} -> {a.base} ok ({len(changed)} files changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
