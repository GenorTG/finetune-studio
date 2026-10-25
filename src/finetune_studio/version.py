"""Release version and build identity.

``VERSION`` at the repo root holds the SemVer of the release line (``MAJOR.MINOR.PATCH``, optionally
``-rc.N`` / ``-beta.N``). Only ``scripts/release.py`` changes it, on a ``release/v*`` or ``hotfix/v*``
branch (CONTRIBUTING.md). A release is the tag ``v<VERSION>`` on ``main``.

``build_version()`` names the exact code: the plain version on a tagged release commit (or an
install without git metadata), else ``<version>+<branch>.<sha8>`` (SemVer build metadata), so a
dev checkout never claims to be a release. Git metadata is read from files, never a subprocess:
this runs in request handlers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<pre>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
_FALLBACK = "0.0.0"


@dataclass(frozen=True)
class GitInfo:
    """What the checkout's ``.git`` says; empty strings when there is no git metadata."""

    branch: str = ""   # "" on a detached HEAD
    commit: str = ""   # full sha
    tags: tuple[str, ...] = ()  # tags that point at ``commit`` (annotated tags peeled)


def release_version(root: Path = REPO_ROOT) -> str:
    """SemVer from ``VERSION``; the installed package metadata when the file is absent (wheel)."""
    try:
        v = (root / "VERSION").read_text(encoding="utf-8").strip()
        if v:
            return v
    except OSError:
        pass
    try:
        from importlib.metadata import version as _dist_version

        return _dist_version("finetune-studio")
    except Exception:  # noqa: BLE001 - not installed: report the fallback, never raise from a version probe
        return _FALLBACK


def is_prerelease(version: str) -> bool:
    """0.x and ``-rc``/``-beta`` versions are pre-releases; 1.0.0 is the first stable release."""
    m = SEMVER_RE.match(version)
    return m is None or m.group("major") == "0" or bool(m.group("pre"))


def release_channel(version: str) -> str:
    """Human label for the UI chip and ``/api/system/version``."""
    return "BETA" if is_prerelease(version) else "STABLE"


def _git_dirs(root: Path) -> tuple[Path, Path] | None:
    """(git dir with HEAD, common dir with refs); a linked worktree's ``.git`` is a file."""
    dot = root / ".git"
    if dot.is_dir():
        return dot, dot
    if dot.is_file():
        text = dot.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir:"):
            return None
        gd = Path(text.split(":", 1)[1].strip())
        gd = gd if gd.is_absolute() else (root / gd).resolve()
        common = gd
        cfile = gd / "commondir"
        if cfile.is_file():
            c = Path(cfile.read_text(encoding="utf-8").strip())
            common = c if c.is_absolute() else (gd / c).resolve()
        return gd, common
    return None


def _packed_refs(common: Path) -> dict[str, str]:
    """ref -> sha from ``packed-refs``; a ``^sha`` line peels the annotated tag above it."""
    out: dict[str, str] = {}
    try:
        lines = (common / "packed-refs").read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    last = ""
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith("^"):
            if last:
                out[last + "^{}"] = ln[1:].strip()
            continue
        sha, _, ref = ln.partition(" ")
        out[ref.strip()] = sha.strip()
        last = ref.strip()
    return out


def _resolve(common: Path, packed: dict[str, str], ref: str) -> str:
    try:
        return (common / ref).read_text(encoding="utf-8").strip()
    except OSError:
        return packed.get(ref, "")


def _peel_tag(common: Path, sha: str) -> str:
    """Commit sha of a loose annotated tag object; loose objects are zlib-compressed."""
    import zlib

    obj = common / "objects" / sha[:2] / sha[2:]
    try:
        raw = zlib.decompress(obj.read_bytes())
    except (OSError, zlib.error):
        return sha  # packed object or lightweight tag: compare as-is
    header, _, body = raw.partition(b"\x00")
    if not header.startswith(b"tag "):
        return sha
    first = body.split(b"\n", 1)[0].decode("ascii", "replace")
    return first.split(" ", 1)[1] if first.startswith("object ") else sha


def git_info(root: Path = REPO_ROOT) -> GitInfo:
    """Branch, commit and the tags on that commit, from ``.git`` files (no subprocess)."""
    try:
        dirs = _git_dirs(root)
        if dirs is None:
            return GitInfo()
        gd, common = dirs
        head = (gd / "HEAD").read_text(encoding="utf-8").strip()
        packed = _packed_refs(common)
        branch = ""
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            branch = ref.removeprefix("refs/heads/")
            commit = _resolve(common, packed, ref)
        else:
            commit = head
        tags: set[str] = set()
        if commit:
            for ref, sha in packed.items():
                if ref.startswith("refs/tags/") and not ref.endswith("^{}"):
                    peeled = packed.get(ref + "^{}", sha)
                    if commit in (sha, peeled):
                        tags.add(ref.removeprefix("refs/tags/"))
            tag_dir = common / "refs" / "tags"
            if tag_dir.is_dir():
                for p in tag_dir.rglob("*"):
                    if p.is_file():
                        sha = p.read_text(encoding="utf-8").strip()
                        if commit in (sha, _peel_tag(common, sha)):
                            tags.add(p.relative_to(tag_dir).as_posix())
        return GitInfo(branch=branch, commit=commit, tags=tuple(sorted(tags)))
    except OSError:
        return GitInfo()


def build_version(root: Path = REPO_ROOT) -> str:
    """``1.2.3`` on the tagged release commit, else ``1.2.3+<branch>.<sha8>`` in a git checkout."""
    v = release_version(root)
    info = git_info(root)
    if not info.commit or f"v{v}" in info.tags:
        return v
    slug = re.sub(r"[^0-9A-Za-z-]+", "-", info.branch or "detached").strip("-") or "detached"
    return f"{v}+{slug}.{info.commit[:8]}"
