"""Post-training export: ensure merged weights, then convert to deployable formats.

Supports raw adapter-only runs by merging onto a compatible 16-bit base at
export time (``base_model`` override or the run's stored base). AWQ was
removed (unmaintained); supported formats are ``gguf``, ``gptq``,
``abliterated``, and ``merged`` (safetensors only).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SUPPORTED_EXPORT_FORMATS: frozenset[str] = frozenset(
    {"gguf", "gptq", "abliterated", "merged"}
)

DEFAULT_GGUF_QUANTS: list[str] = ["f16", "q8_0", "q4_k_m", "q5_k_m"]


def validate_base_model(base_model: str) -> str:
    """Validate a merge-base path or Hub id; return the stripped value.

    Local absolute/relative paths must exist as a directory with
    ``config.json``. Hub-style ``org/repo`` ids are accepted as-is for
    ``resolve_merge_base``. Rejects empty values and null bytes.
    """
    raw = (base_model or "").strip()
    if not raw:
        raise ValueError("base_model is empty")
    if "\x00" in raw:
        raise ValueError("base_model contains invalid characters")

    path = Path(raw)
    # Hub id: org/repo (not an existing local path)
    if "/" in raw and not path.exists() and not os.path.isabs(raw):
        parts = raw.split("/")
        if len(parts) == 2 and all(parts):
            return raw
        raise ValueError(f"invalid Hub model id: {raw!r}")

    if path.exists():
        resolved = path.resolve()
        if not resolved.is_dir():
            raise ValueError(f"base_model is not a directory: {raw}")
        if not (resolved / "config.json").is_file():
            raise ValueError(
                f"base_model directory has no config.json: {resolved}"
            )
        return str(resolved)

    # Non-existent absolute path
    if os.path.isabs(raw):
        raise ValueError(f"base_model path does not exist: {raw}")

    # Bare repo name / relative — let resolve_merge_base search caches
    return raw


def merged_dir_ready(output_path: str) -> bool:
    """True when ``<output_path>/merged/`` has weight files."""
    merged = os.path.join(output_path, "merged")
    if not os.path.isdir(merged):
        return False
    try:
        names = os.listdir(merged)
    except OSError:
        return False
    return any(n.endswith((".safetensors", ".bin")) for n in names)


def adapter_dir_ready(output_path: str) -> bool:
    """True when ``<output_path>/adapter/`` exists and is non-empty."""
    adapter = os.path.join(output_path, "adapter")
    if not os.path.isdir(adapter):
        return False
    try:
        return bool(os.listdir(adapter))
    except OSError:
        return False


def ensure_merged_for_export(
    run: dict[str, Any],
    *,
    base_model: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Ensure ``merged/`` exists for ``run``, merging the adapter if needed.

    ``base_model`` overrides ``run["base_model"]`` after path validation.
    Uses ``merge_adapter_for_run`` / ``resolve_merge_base``.
    """
    output_path = (run.get("output_path") or "").strip()
    if not output_path:
        raise ValueError("run has no output_path")

    if merged_dir_ready(output_path) and not force:
        return {
            "merged_path": os.path.join(output_path, "merged"),
            "skipped": True,
            "merged": True,
        }

    if not adapter_dir_ready(output_path):
        raise ValueError(
            "no merged model and no adapter to merge — "
            "train with merge_on_save or provide an adapter/"
        )

    override = (base_model or "").strip() or None
    if override is not None:
        override = validate_base_model(override)

    run_for_merge = dict(run)
    if override is not None:
        run_for_merge["base_model"] = override

    from finetune_studio.training.engine import merge_adapter_for_run
    from finetune_studio.training.merge_base import MergeBaseNotFound

    try:
        result = merge_adapter_for_run(run_for_merge, force=force)
    except MergeBaseNotFound as e:
        raise ValueError(str(e)) from e

    return {
        "merged_path": result.get("merged_path"),
        "skipped": bool(result.get("skipped")),
        "merged": True,
        "size_bytes": result.get("size_bytes", 0),
        "size_human": result.get("size_human", "0 B"),
    }


def _export_failure(
    error: str,
    *,
    format: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Structured failure payload for sync export (never look like success)."""
    out: dict[str, Any] = dict(extra)
    out.pop("ok", None)
    out["ok"] = False
    out["status"] = "failed"
    out["error"] = error
    if format is not None:
        out["format"] = format
    return out


def export_trained_run(
    run: dict[str, Any],
    *,
    fmt: str = "gguf",
    quants: list[str] | None = None,
    force: bool = False,
    base_model: str | None = None,
) -> dict[str, Any]:
    """Merge if needed, then export ``run`` to ``fmt``.

    Returns a dict with ``ok`` / ``status`` / ``error`` plus format fields.
    Failures always set ``ok=False`` and ``status="failed"``.
    """
    fmt_norm = (fmt or "gguf").strip().lower()
    if fmt_norm == "awq":
        return _export_failure(
            "AWQ export was removed (autoawq unmaintained). "
            "Use format=gptq, gguf, or merged instead.",
            format="awq",
        )
    if fmt_norm not in SUPPORTED_EXPORT_FORMATS:
        return _export_failure(
            f"unknown format: {fmt_norm}",
            format=fmt_norm,
            supported=sorted(SUPPORTED_EXPORT_FORMATS),
        )

    output_path = (run.get("output_path") or "").strip()
    if not output_path:
        return _export_failure("run has no output_path", format=fmt_norm)

    try:
        merge_info = ensure_merged_for_export(
            run, base_model=base_model, force=False,
        )
    except ValueError as e:
        return _export_failure(str(e), format=fmt_norm)

    if fmt_norm == "merged":
        return {
            "ok": True,
            "status": "skipped" if merge_info.get("skipped") else "exported",
            "format": "merged",
            "merged_path": merge_info.get("merged_path"),
            "size_bytes": merge_info.get("size_bytes", 0),
            "size_human": merge_info.get("size_human", "0 B"),
        }

    from finetune_studio.training.engine import TrainingConfig, TrainingEngine

    quant_list = list(quants) if quants else list(DEFAULT_GGUF_QUANTS)
    cfg = TrainingConfig(output_dir=output_path, gguf_quants=quant_list)
    engine = TrainingEngine()
    engine.config = cfg

    if fmt_norm == "gguf":
        gguf_dir = os.path.join(output_path, "gguf")
        if os.path.isdir(gguf_dir) and os.listdir(gguf_dir) and not force:
            return {
                "ok": True,
                "status": "skipped",
                "format": "gguf",
                "gguf_path": gguf_dir,
                "message": (
                    "GGUF already exists. Use force=true to overwrite."
                ),
            }
        result = engine._do_export_gguf(output_path)
        if result.get("error"):
            return _export_failure(
                str(result["error"]), format="gguf", **result
            )
        return {
            "ok": True,
            "status": "exported" if not result.get("skipped") else "skipped",
            "format": "gguf",
            **result,
        }

    if fmt_norm == "abliterated":
        abl_dir = os.path.join(output_path, "abliterated")
        if os.path.isdir(abl_dir) and os.listdir(abl_dir) and not force:
            return {
                "ok": True,
                "status": "skipped",
                "format": "abliterated",
                "message": (
                    "Abliterated model already exists. "
                    "Use force=true to overwrite."
                ),
            }
        result = engine._do_abliteration()
        if result.get("error"):
            return _export_failure(
                str(result["error"]), format="abliterated", **result
            )
        return {"ok": True, "status": "exported", "format": "abliterated", **result}

    # gptq — fail fast when auto_gptq is missing (common host gap)
    from finetune_studio.training.advanced_quant import is_gptq_available

    if not is_gptq_available():
        return _export_failure(
            "No module named 'auto_gptq' — install auto-gptq to export GPTQ, "
            "or choose format=merged / gguf instead.",
            format="gptq",
        )

    result = engine._do_export_gptq(output_path)
    if result.get("error"):
        return _export_failure(str(result["error"]), format="gptq", **result)
    if result.get("skipped"):
        return _export_failure(
            str(result.get("reason", "gptq skipped")),
            format="gptq",
            **result,
        )
    return {"ok": True, "status": "exported", "format": "gptq", **result}
