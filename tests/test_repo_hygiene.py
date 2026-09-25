"""Repo hygiene guard: runtime data must never enter version control.

The dev DB on this box once carried 7,118 project rows, most of them written
by the test suite, and the project corpus dir held 249 orphaned test dirs. This
test makes the "runtime artifacts are never committed" rule from AGENTS.md
enforceable instead of merely documented.

Two failure modes are checked:

1. TRACKED   — a runtime path is already in the index/HEAD.
2. LEAKY     — a runtime path exists on disk but is not ignored, so the next
               ``git add -A`` would commit it.

The ignore side has to be checked against real on-disk paths: a rule like
``/data/*`` plus ``!/data/benchmarks/default.json`` only behaves correctly
because git does not descend into an excluded directory. A rule that looks
right in the file can still leak, so we probe the real matcher via
``git check-ignore`` and the real untracked set via ``git status``.

Skips (rather than fails) when git is unavailable or we are not inside a work
tree, so a source tarball or an odd CI checkout cannot turn this into noise.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Runtime trees that must never be committed. Anchored to the repo root: an
# unanchored `data/` once matched src/finetune_studio/data/ and silently
# ignored new source files, breaking the file-library build on fan-dragon.
RUNTIME_PREFIXES: tuple[str, ...] = (
    "data/",  # except the shipped benchmark seed, checked separately
    "datasets/",  # user corpus files
    "models/",  # downloaded / adapter weights
    "output/",  # run exports
    "projects/",  # per-project curated.db
    "media/",  # inbound media staging
    "screenshots/",
    ".serena/",  # local agent state + generated memories
    ".ruff_cache/",
    ".pytest_cache/",
    ".mypy_cache/",
    "tmp/",
    "rag_data/",
    "uploads/",
    ".venv/",
    "node_modules/",
)

# Files that legitimately live inside a runtime tree and MUST stay committed,
# or the app loses shipped assets (the default benchmark suite is seeded from
# data/benchmarks/default.json).
REQUIRED_TRACKED: tuple[str, ...] = (
    "data/benchmarks/default.json",
)

# Binary/suffix patterns for runtime payloads that must never be tracked
# wherever they appear.
FORBIDDEN_SUFFIXES: tuple[str, ...] = (
    ".db",
    ".sqlite",
    ".sqlite3",
    ".jsonl",
    ".gguf",
    ".safetensors",
)


def _git(*args: str) -> str | None:
    """Run a git command in the repo, returning stdout, or None if unavailable."""
    if shutil.which("git") is None:
        return None
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout


@pytest.fixture(scope="module")
def tracked_files() -> set[str]:
    """Set of repo-relative paths git currently tracks, or skip if unavailable."""
    out = _git("ls-files")
    if out is None:
        pytest.skip("git unavailable or not inside a work tree")
    return {line for line in out.splitlines() if line}


@pytest.fixture(scope="module")
def untracked_files() -> set[str]:
    """Untracked, NOT-ignored paths — exactly what ``git add -A`` would add."""
    out = _git("status", "--porcelain", "--untracked-files=all")
    if out is None:
        pytest.skip("git unavailable or not inside a work tree")
    paths: set[str] = set()
    for line in out.splitlines():
        if not line.startswith("??"):
            continue
        paths.add(line[3:].strip().strip('"'))
    return paths


def test_no_runtime_path_is_tracked(tracked_files: set[str]) -> None:
    """No file inside a runtime tree may be committed."""
    offenders = sorted(
        path
        for path in tracked_files
        if path.startswith(RUNTIME_PREFIXES) and path not in REQUIRED_TRACKED
    )
    assert not offenders, (
        "runtime data is tracked in git — untrack it (`git rm -r --cached <path>`) "
        f"and confirm .gitignore covers it: {offenders}"
    )


def test_no_runtime_payload_suffix_is_tracked(tracked_files: set[str]) -> None:
    """No DB/weights/jsonl payload may be tracked, wherever it sits."""
    offenders = sorted(
        path
        for path in tracked_files
        if path.lower().endswith(FORBIDDEN_SUFFIXES) and path not in REQUIRED_TRACKED
    )
    assert not offenders, f"runtime payload files are tracked in git: {offenders}"


def test_no_unignored_runtime_paths(untracked_files: set[str]) -> None:
    """Nothing untracked-and-unignored may sit in a runtime tree.

    This is the 'always gitignored' half: it fails before the bad ``git add -A``
    happens, rather than after.
    """
    offenders = sorted(
        path
        for path in untracked_files
        if path.startswith(RUNTIME_PREFIXES) and path not in REQUIRED_TRACKED
    )
    assert not offenders, (
        "runtime paths are on disk but NOT gitignored — a `git add -A` would "
        f"commit them: {offenders}"
    )


def test_required_runtime_seeds_are_still_tracked(tracked_files: set[str]) -> None:
    """The ignore rules must not swallow shipped assets the app needs."""
    missing = [path for path in REQUIRED_TRACKED if path not in tracked_files]
    assert not missing, (
        f"shipped assets are no longer tracked (an over-broad .gitignore rule?): {missing}"
    )


def test_ignore_rules_actually_work_on_disk() -> None:
    """Probe ``git check-ignore`` on representative runtime paths.

    Guards the negation trap: `/data/*` excludes the benchmarks directory and
    git will not descend into an excluded directory, so re-including one file
    inside it needs a three-step rule. A single `!/data/benchmarks/` looks
    right in the file but re-exposes the whole runtime hf_cache/ tree.
    """
    must_be_ignored = (
        "data/benchmarks/hf_cache/probe.bin",  # runtime cache inside a re-included dir
        "data/benchmarks/probe.json",  # anything but the shipped seed
        "data/finetune_studio.db",
        "data/projects/probe/datasets/suite.jsonl",
        "datasets/custom/probe.txt",
        "models/probe.gguf",
        "output/probe",
        "projects/probe/curated.db",
        "media/inbound/probe.png",
        ".serena/probe.yml",
    )
    must_be_tracked = REQUIRED_TRACKED

    if _git("rev-parse", "--is-inside-work-tree") is None:
        pytest.skip("not inside a git work tree")

    def _is_ignored(path: str) -> bool:
        # check-ignore exits 0 = ignored, 1 = not ignored. Any other code means
        # git itself failed, which is not this test's business.
        return (
            subprocess.run(
                ["git", "check-ignore", "-q", path],
                cwd=ROOT,
                capture_output=True,
                check=False,
            ).returncode
            == 0
        )

    unignored = [path for path in must_be_ignored if not _is_ignored(path)]
    assert not unignored, (
        f"these runtime paths are NOT ignored — fix .gitignore: {unignored}"
    )

    swallowed = [path for path in must_be_tracked if _is_ignored(path)]
    assert not swallowed, (
        f"shipped assets are now gitignored — the .gitignore negation is too "
        f"broad: {swallowed}"
    )
