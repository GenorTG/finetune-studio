#!/usr/bin/env python3
"""Cut, finish and verify releases (CONTRIBUTING.md "Versions and releases").

    release.py status                      VERSION, last tag, what dev holds since it
    release.py prepare minor|major|patch|X.Y.Z[-rc.N]
                                           from origin/dev: branch release/vX.Y.Z, bump VERSION +
                                           pyproject, date the CHANGELOG section, push, open PR -> main.
                                           On a hotfix/vX.Y.Z branch (cut from main): bump in place,
                                           PR -> main.
    release.py finish X.Y.Z                after the PR is merged: tag vX.Y.Z on origin/main (the
                                           release workflow publishes it), open sync/vX.Y.Z -> dev.
    release.py verify-tag vX.Y.Z           CI: tag, VERSION, pyproject and CHANGELOG agree, tag is on main
    release.py notes X.Y.Z                 print the CHANGELOG section (release notes)

`--no-push` / `--no-pr` keep everything local (tests, dry runs). Stdlib only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from branch_policy import (
    RELEASE_RE,
    SEMVER,
    changelog_has_release,
    is_semver,
    parse_semver,
)

REPO_URL = "https://github.com/GenorTG/finetune-studio"
UNRELEASED = "## [Unreleased]"


class ReleaseError(RuntimeError):
    """A precondition failed; the message says what to do."""


def git(*args: str, check: bool = True) -> str:
    p = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if check and p.returncode != 0:
        raise ReleaseError(f"git {' '.join(args)} failed: {p.stderr.strip() or p.stdout.strip()}")
    return p.stdout.strip()


def gh(*args: str) -> str:
    p = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise ReleaseError(f"gh {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def root() -> Path:
    return Path(git("rev-parse", "--show-toplevel"))


def show(ref: str, path: str) -> str:
    return git("show", f"{ref}:{path}", check=False)


def bump(current: str, how: str) -> str:
    """Next version: major/minor/patch from `current` (its pre-release dropped), or an explicit X.Y.Z."""
    if re.fullmatch(SEMVER, how):
        return how
    core = current.split("-", 1)[0]
    major, minor, patch = (int(x) for x in core.split("."))
    if how == "major":
        return f"{major + 1}.0.0"
    if how == "minor":
        return f"{major}.{minor + 1}.0"
    if how == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ReleaseError(f"bump must be major, minor, patch or X.Y.Z, not {how!r}")


def changelog_section(text: str, version: str) -> str:
    """Body of `## [version]` (or `## [Unreleased]` for version='Unreleased'), without its heading."""
    head = re.compile(rf"^## \[{re.escape(version)}\].*$", re.MULTILINE)
    m = head.search(text)
    if not m:
        return ""
    nxt = re.compile(r"^## \[|^\[[^\]]+\]: ", re.MULTILINE).search(text, m.end())
    return text[m.end(): nxt.start() if nxt else len(text)].strip()


def date_changelog(text: str, version: str, previous: str, today: str) -> str:
    """Move the Unreleased entries under `## [version] - today` and point the compare links."""
    if UNRELEASED not in text:
        raise ReleaseError(f"CHANGELOG.md has no '{UNRELEASED}' heading")
    if changelog_has_release(text, version):
        raise ReleaseError(f"CHANGELOG.md already has a [{version}] section")
    if not changelog_section(text, "Unreleased"):
        raise ReleaseError("CHANGELOG.md's Unreleased section is empty: nothing to release")
    text = text.replace(UNRELEASED, f"{UNRELEASED}\n\n## [{version}] - {today}", 1)
    text = re.sub(r"^\[Unreleased\]: .*$\n?", "", text, flags=re.MULTILINE).rstrip() + "\n"
    links = [f"[Unreleased]: {REPO_URL}/compare/v{version}...dev"]
    if previous and git("rev-parse", "-q", "--verify", f"refs/tags/v{previous}", check=False):
        links.append(f"[{version}]: {REPO_URL}/compare/v{previous}...v{version}")
    else:
        links.append(f"[{version}]: {REPO_URL}/releases/tag/v{version}")
    old = [ln for ln in text.splitlines() if re.match(r"^\[\d[^\]]*\]: ", ln)]
    body = "\n".join(ln for ln in text.splitlines() if ln not in old).rstrip()
    return body + "\n\n" + "\n".join(links + old) + "\n"


def set_pyproject_version(text: str, version: str) -> str:
    new, n = re.subn(r'(?m)^(version\s*=\s*)"[^"]*"', rf'\g<1>"{version}"', text, count=1)
    if n != 1:
        raise ReleaseError("pyproject.toml has no `version = \"...\"` line")
    return new


def require_clean() -> None:
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ReleaseError("the working tree has uncommitted changes; commit or stash them first")


def cmd_status(_: argparse.Namespace) -> int:
    git("fetch", "-q", "--tags", "origin", check=False)
    v = show("origin/main", "VERSION").strip() or "?"
    tags = [t for t in git("tag", "--list", "v*", "--sort=-v:refname").splitlines() if t]
    last = tags[0] if tags else ""
    print(f"main VERSION {v}; last tag {last or '(none)'}")
    rng = f"{last}..origin/dev" if last else "origin/dev"
    log = git("log", "--oneline", "--no-merges", rng, check=False).splitlines()
    print(f"dev holds {len(log)} commit(s) since {last or 'the start'}")
    for ln in log[:20]:
        print(f"  {ln}")
    unreleased = changelog_section(show("origin/dev", "CHANGELOG.md"), "Unreleased")
    print("Unreleased in CHANGELOG (origin/dev):\n" + (unreleased or "  (empty)"))
    return 0


def cmd_prepare(a: argparse.Namespace) -> int:
    require_clean()
    if not a.no_push:
        git("fetch", "-q", "--tags", "origin")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    main_v = show("origin/main", "VERSION").strip() or show("main", "VERSION").strip()
    if not is_semver(main_v):
        main_v = ""  # the old MAJOR.MINOR.PATCH.BUILD file on main: this release is the first SemVer one
    hot = RELEASE_RE.match(branch)
    if hot and hot.group("kind") == "hotfix":
        version = hot.group("version")
        if a.bump not in ("patch", version):
            raise ReleaseError(f"on {branch} the version is {version}; run `release.py prepare patch`")
    else:
        dev_v = show("origin/dev", "VERSION").strip() or show("dev", "VERSION").strip()
        candidates = [v for v in (main_v, dev_v) if is_semver(v)]
        if not candidates:
            raise ReleaseError("neither main nor dev has a SemVer VERSION")
        current = max(candidates, key=parse_semver)
        version = bump(current, a.bump)
        start = "origin/dev" if git("rev-parse", "-q", "--verify", "origin/dev", check=False) else "dev"
        branch = f"release/v{version}"
        git("switch", "-c", branch, start)
    if main_v and parse_semver(version) <= parse_semver(main_v):
        raise ReleaseError(f"{version} is not newer than main's {main_v}")
    r = root()
    previous = main_v
    (r / "VERSION").write_text(version + "\n", encoding="utf-8")
    py = r / "pyproject.toml"
    py.write_text(set_pyproject_version(py.read_text(encoding="utf-8"), version), encoding="utf-8")
    cl = r / "CHANGELOG.md"
    today = a.date or _dt.datetime.now(_dt.UTC).date().isoformat()
    cl.write_text(date_changelog(cl.read_text(encoding="utf-8"), version, previous, today), encoding="utf-8")
    git("add", "VERSION", "pyproject.toml", "CHANGELOG.md")
    git("commit", "-q", "-m", f"release: v{version}")
    print(f"{branch}: VERSION {previous} -> {version}, CHANGELOG dated {today}")
    if a.no_push:
        return 0
    git("push", "-q", "-u", "origin", branch)
    if a.no_pr:
        return 0
    notes = changelog_section(cl.read_text(encoding="utf-8"), version)
    url = gh("pr", "create", "--base", "main", "--head", branch, "--title", f"Release v{version}",
             "--body", f"Release v{version}.\n\n{notes}\n\nAfter merge (merge commit): "
                       f"`make release-finish V={version}`")
    print(f"PR -> main: {url}\nMerge it with a merge commit once CI is green, then: make release-finish V={version}")
    return 0


def cmd_finish(a: argparse.Namespace) -> int:
    v = a.version.removeprefix("v")
    parse_semver(v)
    main_ref = "main" if a.no_push else "origin/main"
    if not a.no_push:
        git("fetch", "-q", "--tags", "origin")
    if show(main_ref, "VERSION").strip() != v:
        raise ReleaseError(f"{main_ref} VERSION is {show(main_ref, 'VERSION').strip()!r}, not {v}: "
                           "merge the release PR first")
    tag = f"v{v}"
    if git("rev-parse", "-q", "--verify", f"refs/tags/{tag}", check=False):
        print(f"tag {tag} exists already")
    else:
        git("tag", "-a", tag, "-m", f"Finetune Studio {tag}", main_ref)
        print(f"tagged {tag} on {main_ref} ({git('rev-parse', '--short', main_ref)})")
        if not a.no_push:
            git("push", "-q", "origin", f"refs/tags/{tag}")
            print("pushed the tag: the release workflow builds and publishes the GitHub Release")
    sync = f"sync/{tag}"
    dev_ref = "dev" if a.no_push else "origin/dev"
    if subprocess.run(["git", "merge-base", "--is-ancestor", main_ref, dev_ref], check=False).returncode == 0:
        print("dev already contains main; no sync needed")
        return 0
    git("switch", "-c", sync, dev_ref)
    p = subprocess.run(["git", "merge", "--no-ff", "-m", f"sync: {tag} back into dev", main_ref],
                       capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise ReleaseError(f"merging {main_ref} into {sync} conflicts; resolve, `git commit`, push {sync} "
                           f"and open a PR into dev (merge commit): {p.stdout.strip()[-400:]}")
    if a.no_push:
        return 0
    git("push", "-q", "-u", "origin", sync)
    if a.no_pr:
        return 0
    url = gh("pr", "create", "--base", "dev", "--head", sync, "--title", f"Sync {tag} into dev",
             "--body", f"Brings the {tag} release commit back into dev. Merge with a merge commit.")
    print(f"PR -> dev: {url}")
    return 0


def cmd_verify_tag(a: argparse.Namespace) -> int:
    tag = a.tag.removeprefix("refs/tags/")
    m = re.fullmatch(rf"v({SEMVER})", tag)
    if not m:
        raise ReleaseError(f"tag {tag!r} is not vX.Y.Z")
    v = m.group(1)
    errors = []
    file_v = show(tag, "VERSION").strip()
    if file_v != v:
        errors.append(f"VERSION at {tag} is {file_v!r}")
    py = re.search(r'(?m)^version\s*=\s*"([^"]*)"', show(tag, "pyproject.toml"))
    if not py or py.group(1) != v:
        errors.append(f"pyproject version at {tag} is {py.group(1) if py else None!r}")
    if not changelog_has_release(show(tag, "CHANGELOG.md"), v):
        errors.append(f"CHANGELOG.md at {tag} has no [{v}] section")
    if subprocess.run(["git", "merge-base", "--is-ancestor", tag, a.main_ref], check=False).returncode != 0:
        errors.append(f"{tag} is not on {a.main_ref}")
    for e in errors:
        print(f"::error::{e}")
    if errors:
        return 1
    print(f"{tag}: VERSION, pyproject, CHANGELOG agree and the tag is on {a.main_ref}")
    return 0


def cmd_notes(a: argparse.Namespace) -> int:
    v = a.version.removeprefix("v")
    text = (root() / "CHANGELOG.md").read_text(encoding="utf-8")
    body = changelog_section(text, v)
    if not body:
        raise ReleaseError(f"CHANGELOG.md has no [{v}] section")
    print(body)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("prepare")
    p.add_argument("bump")
    p.add_argument("--date", help="release date YYYY-MM-DD (default today)")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--no-pr", action="store_true")
    p.set_defaults(fn=cmd_prepare)
    f = sub.add_parser("finish")
    f.add_argument("version")
    f.add_argument("--no-push", action="store_true")
    f.add_argument("--no-pr", action="store_true")
    f.set_defaults(fn=cmd_finish)
    t = sub.add_parser("verify-tag")
    t.add_argument("tag")
    t.add_argument("--main-ref", default="origin/main")
    t.set_defaults(fn=cmd_verify_tag)
    n = sub.add_parser("notes")
    n.add_argument("version")
    n.set_defaults(fn=cmd_notes)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except ReleaseError as e:
        print(f"release: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
