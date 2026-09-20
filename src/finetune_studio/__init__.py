"""finetune_studio — training, evaluation, and deployment for fine-tuned LLMs.

This package is the APPLICATION LAYER for our AI workflow:
  - Train models on custom data (training/)
  - Evaluate models on industry benchmarks (benchmarks/)
  - Compare models side-by-side (compare/)
  - Serve models via a web UI (webui/)
  - Manage RAG documents (rag/)
  - Inspect and validate training data (data/)

It depends on the finetune_studio.templates module for:
  - Jinja2 template rendering (canonical source)
  - Tool-call parsing
  - Built-in tools
"""

_BASE_VERSION = "0.1.0"
__version__ = _BASE_VERSION
# Human-facing maturity label; keep the package version semantic.
__release_channel__ = "EARLY BETA"


def build_version() -> str:
    """Full per-commit build version ``MAJOR.MINOR.PATCH.BUILD``.

    Reads the repo-root ``VERSION`` file (auto-bumped by the pre-commit hook
    so every commit is a new version). Falls back to the base package version
    + ``.0`` when the file is absent (e.g. wheel installs).
    """
    from pathlib import Path

    for cand in (Path(__file__).resolve().parents[2] / "VERSION",
                 Path(__file__).resolve().parents[1] / "VERSION"):
        try:
            v = cand.read_text(encoding="utf-8").strip()
            if v:
                return v
        except OSError:
            pass
    return f"{_BASE_VERSION}.0"


# Every consumer of __version__ (UI chip, /api/system/version) gets the
# per-commit build string; pyproject metadata keeps the base semver.
__version__ = build_version()
