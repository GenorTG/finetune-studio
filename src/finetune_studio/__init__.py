"""finetune_studio — training, evaluation, and deployment for fine-tuned LLMs.

This package is the APPLICATION LAYER for our AI workflow:
  - Train models on custom data (training/)
  - Evaluate models on industry benchmarks (benchmarks/)
  - Compare models side-by-side (compare/)
  - Serve models via a web UI (webui/)
  - Smart Brain retrieval (smart_brain/)
  - Inspect and validate training data (data/)

It depends on the finetune_studio.templates module for:
  - Jinja2 template rendering (canonical source)
  - Tool-call parsing
  - Built-in tools
"""

# GPU visibility policy (FTS_GPU_DEVICES / FTS_GPU_EXCLUDE) must be in the
# environment before torch or llama.cpp initialise a driver anywhere in-process.
from finetune_studio.accel.env import (
    apply_device_policy as _apply_device_policy,
)

_apply_device_policy()

# Versioning (CONTRIBUTING.md "Versions and releases"): VERSION holds the SemVer of the release line,
# changed only by scripts/release.py; a release is the tag v<VERSION> on main.
from finetune_studio.version import (  # after the device policy on purpose
    build_version,
    release_channel,
    release_version,
)

__release__ = release_version()
# The exact code: "1.2.3" on a tagged release, "1.2.3+dev.<sha8>" in a dev checkout. The UI chip and
# /api/system/version show this; pyproject's version is __release__.
__version__ = build_version()
# Human-facing maturity label: BETA for 0.x and -rc/-beta versions, STABLE from 1.0.0.
__release_channel__ = release_channel(__release__)

__all__ = ["__release__", "__release_channel__", "__version__", "build_version", "release_version"]
