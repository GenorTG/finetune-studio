"""Resolve a non-quantized (16-bit) base model for PEFT merge after QLoRA.

QLoRA trains against a bitsandbytes/nf4 base. Merging that adapter in-place
and calling ``save_pretrained`` trips transformers' ``revert_weight_conversion``
(``NotImplementedError``). The working recipe is: load the 16-bit sibling base
in bf16, ``PeftModel.from_pretrained`` + ``merge_and_unload``, then save.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Strip longest suffixes first.
_QUANT_SUFFIXES: tuple[str, ...] = (
    "-unsloth-bnb-4bit",
    "-bnb-4bit",
    "-4bit",
)

_PREFERRED_ORGS: tuple[str, ...] = ("unsloth", "Qwen", "qwen")


class MergeBaseNotFound(Exception):
    """No local non-quantized sibling of a quantized training base was found."""


def _hub_cache_root() -> Path:
    """HuggingFace hub cache directory (``~/.cache/huggingface/hub`` by default)."""
    env_hub = os.environ.get("HF_HUB_CACHE")
    if env_hub:
        return Path(env_hub)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _app_hf_models_root() -> Path:
    """App-local HF download dir used by ``/api/hf/local``."""
    return Path.home() / ".finetune-studio" / "hf_models"


def _read_config_dict(model_path: str | Path) -> dict | None:
    """Load ``config.json`` from a local dir or return None if missing."""
    path = Path(model_path)
    if path.is_file() and path.name == "config.json":
        cfg_path = path
    elif path.is_dir():
        cfg_path = path / "config.json"
    else:
        return None
    if not cfg_path.is_file():
        return None
    try:
        with open(cfg_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def is_quantized_config(config: dict | None) -> bool:
    """True if ``config`` looks like bitsandbytes / nf4 / 4-bit."""
    if not config:
        return False
    qc = config.get("quantization_config")
    if not isinstance(qc, dict):
        return False
    method = str(qc.get("quant_method", "")).lower()
    if method in ("bitsandbytes", "bnb", "bnb-4bit"):
        return True
    if qc.get("load_in_4bit") is True:
        return True
    qtype = str(qc.get("bnb_4bit_quant_type", "")).lower()
    if qtype in ("nf4", "fp4"):
        return True
    blob = json.dumps(qc).lower()
    return "nf4" in blob or "bitsandbytes" in blob


def strip_quant_suffixes(repo: str) -> str:
    """Strip ``-unsloth-bnb-4bit`` / ``-bnb-4bit`` / ``-4bit`` from a repo name."""
    lower = repo.lower()
    for suf in _QUANT_SUFFIXES:
        if lower.endswith(suf):
            return repo[: len(repo) - len(suf)]
    return repo


def _parse_base_identity(base_model: str) -> tuple[str | None, str]:
    """Return ``(org_or_None, bare_repo)`` derived from a path or HF id."""
    raw = base_model.rstrip("/\\")
    # HF hub cache: .../models--Org--Repo/snapshots/<hash>
    m = re.search(r"models--([^/\\]+)--([^/\\]+)", raw)
    if m:
        return m.group(1), strip_quant_suffixes(m.group(2))
    # App download: Org__Repo
    base_name = Path(raw).name
    if "__" in base_name and (Path(raw).is_dir() or "/" not in raw.replace("\\", "/")):
        org, repo = base_name.split("__", 1)
        return org, strip_quant_suffixes(repo)
    # Hub repo id: org/repo
    if "/" in raw and not os.path.isabs(raw) and not Path(raw).exists():
        org, repo = raw.split("/", 1)
        return org, strip_quant_suffixes(repo)
    # Local directory path — use leaf name
    leaf = Path(raw).name
    if "__" in leaf:
        org, repo = leaf.split("__", 1)
        return org, strip_quant_suffixes(repo)
    return None, strip_quant_suffixes(leaf)


def _snapshot_is_complete(snap: Path) -> bool:
    """Snapshot must have config.json and at least one weight file."""
    if not (snap / "config.json").is_file():
        return False
    if is_quantized_config(_read_config_dict(snap)):
        return False
    for pattern in ("*.safetensors", "*.bin"):
        if any(snap.glob(pattern)):
            return True
    return False


def _best_snapshot_in_repo_dir(repo_dir: Path) -> Path | None:
    """Pick a snapshots/<hash> under a HF cache repo dir, or the dir itself."""
    snaps = repo_dir / "snapshots"
    if snaps.is_dir():
        candidates = sorted(
            (p for p in snaps.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for snap in candidates:
            if _snapshot_is_complete(snap):
                return snap
        return None
    if _snapshot_is_complete(repo_dir):
        return repo_dir
    return None


def _find_in_hub_cache(bare_repo: str, preferred_orgs: tuple[str, ...] | None = None) -> str | None:
    """Search ``~/.cache/huggingface/hub/models--*`` for a matching non-quant base."""
    hub = _hub_cache_root()
    if not hub.is_dir():
        return None
    bare_l = bare_repo.lower()
    preferred = [o.lower() for o in (preferred_orgs or ())]

    def score(org: str, repo: str) -> tuple[int, str]:
        # Lower is better: preferred org first, then any exact repo match.
        org_l = org.lower()
        pref_rank = preferred.index(org_l) if org_l in preferred else 100
        return (pref_rank, org_l)

    hits: list[tuple[tuple[int, str], Path]] = []
    for entry in hub.iterdir():
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        parts = entry.name.split("--", 2)
        if len(parts) != 3:
            continue
        _prefix, org, repo = parts
        if repo.lower() != bare_l:
            continue
        snap = _best_snapshot_in_repo_dir(entry)
        if snap is None:
            continue
        hits.append((score(org, repo), snap))
    if not hits:
        return None
    hits.sort(key=lambda t: t[0])
    return str(hits[0][1])


def _find_in_app_hf_models(bare_repo: str, preferred_orgs: tuple[str, ...] | None = None) -> str | None:
    """Search ``~/.finetune-studio/hf_models/<Org>__<Repo>``."""
    root = _app_hf_models_root()
    if not root.is_dir():
        return None
    bare_l = bare_repo.lower()
    preferred = [o.lower() for o in (preferred_orgs or ())]

    hits: list[tuple[tuple[int, str], Path]] = []
    for entry in root.iterdir():
        if not entry.is_dir() or "__" not in entry.name:
            continue
        org, repo = entry.name.split("__", 1)
        if repo.lower() != bare_l:
            continue
        # Prefer a snapshots/ child if present, else the dir itself.
        snap = _best_snapshot_in_repo_dir(entry)
        if snap is None:
            continue
        org_l = org.lower()
        pref_rank = preferred.index(org_l) if org_l in preferred else 100
        hits.append(((pref_rank, org_l), snap))
    if not hits:
        return None
    hits.sort(key=lambda t: t[0])
    return str(hits[0][1])


def _suggested_16bit_repo(bare_repo: str) -> str:
    """Human-facing Hub id to download when nothing is local."""
    # Qwen3-style: capitalize segments while keeping dots (qwen3-0.6b → Qwen3-0.6B).
    if bare_repo.lower().startswith("qwen"):
        parts = bare_repo.split("-")
        pretty: list[str] = []
        for i, p in enumerate(parts):
            if i == 0:
                pretty.append(p[0].upper() + p[1:] if p else p)
            elif p.replace(".", "", 1).isdigit() or (p and p[0].isdigit()):
                # 0.6b → 0.6B
                pretty.append(p[:-1] + p[-1].upper() if p[-1:].isalpha() else p)
            else:
                pretty.append(p)
        return f"Qwen/{'-'.join(pretty)}"
    return f"Qwen/{bare_repo}"


def resolve_merge_base(base_model: str) -> str:
    """Return a local path or id suitable for a 16-bit PEFT merge.

    If ``base_model`` has no bitsandbytes/nf4 ``quantization_config``, it is
    returned unchanged. Otherwise derive the 16-bit sibling repo name and search
    the HF hub cache and the app HF download dir for a matching non-quantized
    snapshot. Raises ``MergeBaseNotFound`` when nothing local is available.
    """
    if not base_model or not str(base_model).strip():
        raise MergeBaseNotFound("base_model is empty; cannot resolve a merge base")

    base = str(base_model).strip()
    local_cfg = _read_config_dict(base)
    # Remote / unresolved id with no local config: treat as possibly quantized
    # only when the id itself carries a quant suffix; otherwise pass through.
    if local_cfg is not None and not is_quantized_config(local_cfg):
        return base
    if local_cfg is None:
        # No local config — if the name doesn't look quantized, keep as-is
        # (HF will fetch / transformers will load).
        _org, bare_probe = _parse_base_identity(base)
        if bare_probe == Path(base).name or (
            "/" in base and strip_quant_suffixes(base.split("/", 1)[-1]) == base.split("/", 1)[-1]
        ):
            # No quant suffix stripped → not a quantized id
            if "/" in base and not Path(base).exists():
                leaf = base.split("/", 1)[-1]
                if strip_quant_suffixes(leaf) == leaf:
                    return base
            elif not any(base.lower().endswith(s) for s in _QUANT_SUFFIXES):
                return base

    org, bare = _parse_base_identity(base)
    preferred: list[str] = list(_PREFERRED_ORGS)
    if org and org not in preferred:
        preferred.insert(0, org)

    for finder in (_find_in_hub_cache, _find_in_app_hf_models):
        hit = finder(bare, tuple(preferred))
        if hit:
            return hit

    suggested = _suggested_16bit_repo(bare)
    raise MergeBaseNotFound(
        f"No local 16-bit base found for quantized model {base!r}. "
        f"Download {suggested} (or unsloth/{bare}) on the HF models page, "
        f"then re-run merge."
    )
