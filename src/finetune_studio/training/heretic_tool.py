"""Heretic — the optional, isolated censorship-removal (abliteration) tool.

WHY AN EXTERNAL PROGRAM AND NOT VENDORED CODE
=============================================
Heretic (https://github.com/p-e-w/heretic) is AGPL-3.0-or-later. Finetune Studio
ships under its own license, so copying Heretic's source into this tree would put
the whole product under the AGPL. The tool is therefore installed as an
*unmodified external program* in its own virtual environment and driven through
its command line. Nothing from Heretic's source is imported or adapted here.

THE OVERLAY VENV: ONE TORCH, ~650 MB
====================================
A plain ``pip install heretic-llm`` pulls a *second* PyTorch (~6 GB, another CUDA
build) and every ML package again. Instead the overlay venv gets a
``fts-shared.pth`` that appends this app's ``site-packages`` to its path, so
torch, transformers, peft, datasets, accelerate, bitsandbytes and numpy resolve to
*our* install and only Heretic's own missing packages are added on top.

WHY THE PIN IS A COMMIT AND NOT THE PYPI RELEASE
================================================
1.4.0 (PyPI) cannot run unattended: its save step is a bare ``questionary`` prompt
with no settings field behind it, so a headless drag abliteration would block on
stdin forever. The headless flags (``--model-action``, ``--save-directory``,
``--export-strategy``, ``--checkpoint-action``, ``--trial-index``) exist only in
the 2.0 development line. :data:`PINNED_REF` is that line, and
:func:`run_abliteration` passes every one of those flags explicitly — a missing
flag is a hang, not a nuisance, which is why :func:`missing_headless_flags`
refuses to start rather than hoping.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/p-e-w/heretic"
GITHUB_API = "https://api.github.com/repos/p-e-w/heretic"

#: The 2.0 development line: the first line whose CLI can save without a TTY.
PINNED_REF = "d34dd9192da06d203eb7e80d20229117bb4e981c"
#: Human label for :data:`PINNED_REF` (shown in Settings).
PINNED_LABEL = "2.0.0.dev0 @ d34dd91 (2026-10-26)"

#: Distribution name in the overlay venv.
PACKAGE = "heretic-llm"

#: Heretic's own dependencies that this app does not already install.
#: Everything else (torch, transformers, peft, datasets, accelerate,
#: bitsandbytes, huggingface-hub, numpy, safetensors, pydantic, tqdm,
#: psutil, rich, pydantic-settings) comes from the app venv through the .pth.
#:
#: ``lm-eval`` is installed WITHOUT its ``[hf]`` extra on purpose. That extra only
#: pins ``transformers``/``torch``/``accelerate``/``peft`` — every one of which the
#: app already provides — and pip/uv cannot see them (the overlay is a .pth, not a
#: real dependency), so the extra drags in a second PyTorch, triton and ~3 GB of
#: CUDA libraries (5.8 GB instead of ~0.65 GB). ``lm_eval.models.huggingface``
#: imports fine against our torch without it; ``tests/test_heretic_tool.py`` fails
#: if this list ever reintroduces a duplicated heavyweight.
EXTRA_DEPS: tuple[str, ...] = (
    "optuna>=5.0,<6",
    "lm-eval>=0.4,<0.5",
    "questionary>=2.1,<3",
    "immutabledict>=4.3,<5",
    "langdetect>=1.0,<2",
    "tomli-w>=1.2,<2",
    "py-cpuinfo>=9.0,<10",
)

#: Packages that must come from the app venv, never be installed into the overlay:
#: a second copy means a second CUDA stack and a multi-GB install. Light library
#: dependencies (``datasets`` ~3 MB, ``numpy`` ~30 MB, resolved by ``lm-eval``
#: itself) are NOT listed: duplicating those is harmless and pinning them away
#: would only fight the resolver.
SHARED_PACKAGES: tuple[str, ...] = (
    "torch", "torchvision", "triton", "transformers", "peft", "accelerate",
    "nvidia", "bitsandbytes",
)

#: Whole-run ceiling; a 100-trial abliteration of a 9B model is hours, not minutes.
DEFAULT_TIMEOUT_S = 6 * 3600.0
POLL_HEARTBEAT_S = 2.0

#: Flags without which the child would block on an interactive prompt.
_REQUIRED_FLAGS: tuple[str, ...] = (
    "--model-action",
    "--save-directory",
    "--export-strategy",
    "--checkpoint-action",
    "--trial-index",
)

#: Fallback launcher for ``python -c``: what the console script does, without the
#: script. Only used when the venv has no ``bin/heretic`` (a hand-built install).
_BORING_MAIN = "from heretic.main import main; raise SystemExit(main())"


class HereticError(RuntimeError):
    """The tool is missing, broken, or refused to run."""


@dataclass(frozen=True)
class HereticStatus:
    """What Settings, the Export page and ``fts heretic status`` show."""

    installed: bool
    version: str = ""
    ref: str = ""
    python: str = ""
    root: str = ""
    hint: str = ""
    pinned: str = PINNED_LABEL
    missing_flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def usable(self) -> bool:
        """Installed *and* capable of running without a terminal."""
        return self.installed and not self.missing_flags

    def as_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "usable": self.usable,
            "version": self.version,
            "ref": self.ref,
            "python": self.python,
            "root": self.root,
            "hint": self.hint,
            "pinned": self.pinned,
            "missing_flags": list(self.missing_flags),
        }


# ── locations ────────────────────────────────────────────────────────────────

def root() -> Path:
    """``$FTS_ROOT/heretic`` — the checkout + overlay venv live here."""
    env = os.environ.get("FTS_HERETIC_ROOT")
    if env:
        return Path(env)
    from finetune_studio.access import paths as access_paths
    return access_paths.root() / "heretic"


def checkout_dir() -> Path:
    return root() / "src"


def venv_dir() -> Path:
    return root() / "venv"


def python_path() -> Path:
    return venv_dir() / "bin" / "python"


def overlay_site_packages() -> Path:
    """The overlay venv's own site-packages (never the shared app packages)."""
    return venv_dir() / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def overlay_duplicates() -> tuple[str, ...]:
    """Heavyweight packages copied into the overlay instead of shared with the app.

    Any entry here means the install carries its own CUDA stack (seconds of extra
    disk per package, ~5 GB in total). Read straight off the filesystem: what a
    resolver promised is not evidence, what is on disk is.
    """
    site = overlay_site_packages()
    if not site.is_dir():
        return ()
    return tuple(p for p in SHARED_PACKAGES if (site / p).exists())


def app_site_packages() -> Path:
    """This app's site-packages: the half of the overlay that is not duplicated."""
    import sysconfig
    return Path(sysconfig.get_paths()["purelib"])


# ── probing ──────────────────────────────────────────────────────────────────

def _run(cmd: list[str], *, timeout: float = 120.0, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False,
        env={**os.environ, **(env or {})},
    )


def installed_version(py: Path | None = None) -> str:
    """Installed ``heretic-llm`` version, or ``""`` when absent."""
    py = py or python_path()
    if not py.exists():
        return ""
    try:
        out = _run([str(py), "-c",
                    "import importlib.metadata as m; print(m.version('heretic-llm'))"], timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def installed_ref() -> str:
    """The git commit of the checkout the overlay was installed from, or ``""``."""
    src = checkout_dir()
    if not (src / ".git").exists():
        return ""
    try:
        out = _run(["git", "-C", str(src), "rev-parse", "HEAD"], timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def entry_point() -> Path:
    """Heretic's console script, which is the *only* working entry point.

    ``python -m heretic.main`` looks like it should work and does not: the module
    defines ``main()`` but has no ``if __name__ == "__main__"`` block, so it is
    imported and exits 0 having done nothing — a silent no-op that reports
    success and writes no model. The installed console script calls ``main()``.
    """
    return venv_dir() / "bin" / "heretic"


def missing_headless_flags(py: Path | None = None) -> tuple[str, ...]:
    """Flags the installed build does not offer; non-empty means "cannot run headless".

    Checked against the real ``--help`` rather than a version string: a source
    change can drop a flag inside the same version, and a missing flag turns a
    batch abliteration into a job that hangs until its timeout.
    """
    py = py or python_path()
    if not py.exists():
        return ()
    entry = entry_point()
    cmd = [str(entry), "--help"] if entry.exists() else [str(py), "-c", _BORING_MAIN, "--help"]
    try:
        out = _run(cmd, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return _REQUIRED_FLAGS
    text = out.stdout + out.stderr
    if "--model-action" not in text:
        # Help can fail before parsing on a broken build: report all, be explicit.
        return _REQUIRED_FLAGS
    return tuple(f for f in _REQUIRED_FLAGS if f not in text)


def probe() -> HereticStatus:
    """Current state of the optional tool (never raises)."""
    py = python_path()
    if not py.exists():
        return HereticStatus(
            installed=False, python=str(py), root=str(root()),
            hint="Heretic is not installed. Install it from Settings → Heretic, or run: fts heretic install",
        )
    version = installed_version(py)
    if not version:
        return HereticStatus(
            installed=False, python=str(py), root=str(root()),
            hint="The Heretic environment exists but the package is not importable. Re-install it (fts heretic install).",
        )
    missing = missing_headless_flags(py)
    hint = ""
    if missing:
        hint = (
            "This Heretic build cannot run unattended (it is missing "
            + ", ".join(missing)
            + "). Re-install from the pinned 2.0 development line: fts heretic install"
        )
    return HereticStatus(
        installed=True, version=version, ref=installed_ref(), python=str(py),
        root=str(root()), hint=hint, missing_flags=missing,
    )


# ── upstream release tracking ────────────────────────────────────────────────

def _github(path: str, timeout: float = 20.0) -> Any:
    req = urllib.request.Request(
        f"{GITHUB_API}{path}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "finetune-studio"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@dataclass(frozen=True)
class UpstreamInfo:
    """What GitHub says about Heretic right now."""

    latest_release: str = ""
    latest_release_at: str = ""
    latest_release_url: str = ""
    head_commit: str = ""
    head_at: str = ""
    head_message: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "latest_release": self.latest_release,
            "latest_release_at": self.latest_release_at,
            "latest_release_url": self.latest_release_url,
            "head_commit": self.head_commit,
            "head_at": self.head_at,
            "head_message": self.head_message,
            "error": self.error,
        }


def check_upstream() -> UpstreamInfo:
    """Read Heretic's newest release tag and the newest commit on its default branch.

    Best-effort: a network failure returns ``error`` instead of raising, so the
    Settings card can say "could not check" without breaking the page.
    """
    try:
        release = _github("/releases/latest")
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError, TimeoutError) as e:
        return UpstreamInfo(error=f"GitHub unreachable: {type(e).__name__}: {e}")
    try:
        commit = _github("/commits?per_page=1")
        commit = commit[0] if isinstance(commit, list) and commit else {}
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError, TimeoutError):
        commit = {}
    return UpstreamInfo(
        latest_release=str(release.get("tag_name") or ""),
        latest_release_at=str(release.get("published_at") or ""),
        latest_release_url=str(release.get("html_url") or ""),
        head_commit=str(commit.get("sha") or ""),
        head_at=str((commit.get("commit", {}).get("committer", {}) or {}).get("date") or ""),
        head_message=str(commit.get("commit", {}).get("message") or "").splitlines()[0][:120],
    )


def update_available(status: HereticStatus | None = None, upstream: UpstreamInfo | None = None) -> bool:
    """True when upstream has a commit this install does not have."""
    status = status or probe()
    upstream = upstream or check_upstream()
    if upstream.error or not upstream.head_commit:
        return False
    if not status.installed:
        return True
    return status.ref != upstream.head_commit


# ── install / update ─────────────────────────────────────────────────────────

def _uv() -> str:
    uv = shutil.which("uv")
    if not uv:
        raise HereticError(
            "uv is required to install Heretic (the app installs through uv). "
            "Install uv, or run install.sh, then retry."
        )
    return uv


def _run_logged(cmd: list[str], *, on_line: Any = None, env: dict[str, str] | None = None,
                cwd: str | None = None, timeout: float = 1800.0) -> None:
    """Run ``cmd`` streaming its output to the log and ``on_line``."""
    log.info("heretic: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, **(env or {})}, cwd=cwd,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log.info("heretic| %s", line)
                if on_line:
                    on_line(line)
        proc.wait(timeout=timeout)
    finally:
        if proc.poll() is None:
            proc.kill()
    if proc.returncode != 0:
        raise HereticError(f"{os.path.basename(cmd[0])} failed (rc={proc.returncode}); see the log")


def _clone_or_fetch(ref: str, on_line: Any = None) -> None:
    """Shallow clone Heretic at ``ref`` (re-clones when the ref or the state differs)."""
    src = checkout_dir()
    if (src / ".git").exists():
        _run_logged(["git", "-C", str(src), "fetch", "--depth", "1", "origin", ref], on_line=on_line)
        _run_logged(["git", "-C", str(src), "checkout", "--detach", "FETCH_HEAD"], on_line=on_line)
        return
    shutil.rmtree(src, ignore_errors=True)
    src.parent.mkdir(parents=True, exist_ok=True)
    _run_logged(["git", "clone", "--filter=blob:none", REPO_URL, str(src)], on_line=on_line)
    _run_logged(["git", "-C", str(src), "checkout", "--detach", ref], on_line=on_line)


def _make_venv(on_line: Any = None) -> None:
    """Overlay venv: fresh Python + a ``.pth`` pointing at this app's packages.

    ``--system-site-packages`` is *not* used: it copies the base interpreter's
    path, and when the app itself runs from a venv that path contributes nothing,
    so Heretic would see no torch at all.
    """
    venv = venv_dir()
    shutil.rmtree(venv, ignore_errors=True)
    _run_logged([_uv(), "venv", "--python", os.sys.executable, str(venv)], on_line=on_line)
    site = Path(_run([str(python_path()), "-c",
                      "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                     timeout=60).stdout.strip())
    site.mkdir(parents=True, exist_ok=True)
    (site / "fts-shared.pth").write_text(str(app_site_packages()) + "\n", encoding="utf-8")


def install(ref: str = PINNED_REF, *, on_line: Any = None) -> HereticStatus:
    """Clone, build the overlay venv and install Heretic's own dependencies.

    Idempotent: re-running rebuilds the venv from scratch, which is also the
    repair path. Costs ~650 MB of downloads, not the ~6 GB a plain install would.
    """
    uv = _uv()
    root().mkdir(parents=True, exist_ok=True)
    on_line = on_line or (lambda _l: None)
    on_line(f"fetching {REPO_URL} at {ref[:7]}")
    _clone_or_fetch(ref, on_line=on_line)
    on_line("building the overlay environment (reuses this app's torch)")
    _make_venv(on_line=on_line)
    on_line("installing heretic-llm without its dependency pins")
    _run_logged([uv, "pip", "install", "--python", str(python_path()), "--no-deps",
                 "-e", str(checkout_dir())], on_line=on_line)
    on_line("installing Heretic's own packages (" + ", ".join(d.split(">")[0] for d in EXTRA_DEPS) + ")")
    _run_logged([uv, "pip", "install", "--python", str(python_path()), *EXTRA_DEPS], on_line=on_line)
    status = probe()
    if not status.installed:
        raise HereticError(f"Heretic did not install correctly: {status.hint}")
    if status.missing_flags:
        raise HereticError(
            "Installed Heretic cannot run unattended (missing " + ", ".join(status.missing_flags) + ")"
        )
    dupes = overlay_duplicates()
    if dupes:
        # The overlay exists to share one CUDA stack. A duplicate here means the
        # install silently grew by gigabytes, so say so instead of hiding it.
        raise HereticError(
            "The Heretic environment duplicated packages that should be shared with this app ("
            + ", ".join(dupes)
            + "). Re-install after removing the offending extra; the overlay must not carry its own torch."
        )
    on_line(f"Heretic {status.version} ready")
    return status


def uninstall() -> None:
    """Remove the checkout and the overlay venv (never touches the app's own env)."""
    shutil.rmtree(root(), ignore_errors=True)


# ── running ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AbliterationRun:
    """One headless Heretic run, fully specified (nothing is asked on stdin)."""

    model_path: str
    output_dir: str
    n_trials: int = 40
    n_startup_trials: int = 12
    seed: int = 42
    checkpoint_dir: str = ""
    quantization: str = "none"
    system_prompt: str = "You are a helpful assistant."


def build_command(run: AbliterationRun) -> list[str]:
    """The exact argv for one unattended run (every prompt closed by a flag)."""
    entry = entry_point()
    # The console script when it exists (the module has no __main__ guard, so
    # ``-m heretic.main`` is a silent no-op); ``-c`` is the equivalent fallback.
    launcher = [str(entry)] if entry.exists() else [str(python_path()), "-c", _BORING_MAIN]
    cmd = [
        *launcher,
        "--model", run.model_path,
        # Save the merged full model, then leave: no menu, no path prompt.
        "--model-action", "save",
        "--save-directory", run.output_dir,
        "--export-strategy", "merge",
        # A leftover Optuna study must not stop a batch job with a question.
        "--checkpoint-action", "restart",
        "--trial-index", "0",
        "--n-trials", str(run.n_trials),
        "--n-startup-trials", str(run.n_startup_trials),
        "--seed", str(run.seed),
        "--quantization", run.quantization,
        "--system-prompt", run.system_prompt,
    ]
    if run.checkpoint_dir:
        cmd += ["--study-checkpoint-dir", run.checkpoint_dir]
    return cmd


def run_env() -> dict[str, str]:
    """Child environment: the parent's, plus this app's GPU pin and no pager.

    Built ON TOP of ``os.environ`` on purpose: ``proc_runner`` hands this dict
    straight to ``Popen``, which *replaces* the child's environment. A literal
    partial dict therefore dropped the parent's ``CUDA_VISIBLE_DEVICES`` mask, and
    Heretic saw both cards (including the GTX 1070, which this PyTorch has no
    kernels for) and died with ``no kernel image is available``.
    """
    from finetune_studio import accel

    env = dict(os.environ)
    env.update({
        "TERM": "dumb",
        "NO_COLOR": "1",
        "PYTHONUNBUFFERED": "1",
        # Heretic is an external tool: never let it write into our package tree.
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    try:
        env.update(accel.isolated_env(accel.get_accelerator(), env))
    except Exception:  # accelerator probing must not block an export
        log.debug("heretic: could not pin the GPU", exc_info=True)
    return env


def run_abliteration(
    run: AbliterationRun,
    *,
    cancelled: Any = None,
    on_tick: Any = None,
    on_start: Any = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run Heretic once and return what it wrote.

    Raises :class:`HereticError` when the tool is unusable, and the runner's own
    exceptions on failure/cancel/timeout (their whole process group is killed).
    """
    from finetune_studio.training.proc_runner import run_group_subprocess

    status = probe()
    if not status.installed:
        raise HereticError(status.hint or "Heretic is not installed")
    if status.missing_flags:
        raise HereticError(status.hint or "Heretic cannot run unattended")

    model = Path(run.model_path)
    if not (model / "config.json").is_file():
        raise HereticError(
            f"{model} is not a loadable model directory (no config.json). "
            "Merge the run first; Heretic ablates full models, not loose adapters."
        )
    out = Path(run.output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_command(run)
    log.info("heretic run: %s", " ".join(cmd))
    run_group_subprocess(
        cmd,
        timeout=timeout,
        cancelled=cancelled,
        on_tick=on_tick,
        on_start=on_start,
        env=run_env(),
    )
    return inspect_output(str(out))


def inspect_output(output_dir: str) -> dict[str, Any]:
    """Verify a finished run wrote a loadable model, and size it.

    A directory of shards without ``config.json`` is not a model: every loader in
    the app refuses it. This is the check the previous abliteration export lacked,
    so it fails here instead of producing an export that only looks finished.
    """
    out = Path(output_dir)
    if not out.is_dir():
        raise HereticError(f"Heretic reported success but {out} does not exist")
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    weights = [f for f in files if f.endswith((".safetensors", ".bin"))]
    has_config = "config.json" in files
    has_tokenizer = any(f in files for f in ("tokenizer.json", "tokenizer_config.json",
                                             "tokenizer.model", "vocab.json"))
    if not weights:
        raise HereticError(f"Heretic wrote no weight files into {out}")
    if not has_config:
        raise HereticError(
            f"Heretic wrote weights but no config.json into {out}; the result would not load. "
            "Re-run the export (a partial save is discarded, not reported as done)."
        )
    if not has_tokenizer:
        raise HereticError(
            f"Heretic wrote a model without tokenizer files into {out}; it would not load for chat or testing."
        )
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    return {
        "output_path": str(out),
        "size_bytes": size,
        "files": files,
        "formats": sorted({Path(f).suffix.lstrip(".") for f in files if "." in f}),
    }
