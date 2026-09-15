"""Post-training export: ensure merged weights, then convert to deployable formats.

Supports raw adapter-only runs by merging onto a compatible 16-bit base at
export time (``base_model`` override or the run's stored base). AWQ was
removed (unmaintained); supported formats are ``gguf``, ``gptq``,
``abliterated``, and ``merged`` (safetensors only).

GGUF exports only report success when the requested ``.gguf`` artifact(s)
exist on disk and are non-empty — never on HTTP 200 / empty dirs alone.
GPTQ success requires a non-empty quantized directory (config + weights).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from finetune_studio.training.gguf_convert import (  # noqa: F401 — re-export
    GGUF_CONVERTER_MISSING_MSG,
    convert_merged_to_gguf,
    find_gguf_convert_script,
    find_llama_quantize,
    gguf_filename_matches_quant,
    llama_cpp_search_paths,
    normalize_gguf_quant,
    verify_gguf_artifacts,
)

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


def _export_gguf(
    engine: Any,
    *,
    output_path: str,
    quant_list: list[str],
    force: bool,
) -> dict[str, Any]:
    """Run GGUF conversion and only succeed when artifacts verify non-empty."""
    gguf_dir = os.path.join(output_path, "gguf")
    existing = verify_gguf_artifacts(gguf_dir, quant_list)
    if existing["ok"] and not force:
        return {
            "ok": True,
            "status": "skipped",
            "format": "gguf",
            "gguf_path": gguf_dir,
            "files": existing["files"],
            "quants": quant_list,
            "message": (
                "GGUF already exists. Use force=true to overwrite."
            ),
        }

    if (
        find_gguf_convert_script() is None
        and os.environ.get("FTS_SKIP_EXPORT") != "1"
    ):
        return _export_failure(
            GGUF_CONVERTER_MISSING_MSG,
            format="gguf",
            gguf_path=gguf_dir,
            quants=quant_list,
            missing=quant_list,
        )

    # TrainingEngine._do_export_gguf uses convert_merged_to_gguf; tests may patch it.
    try:
        result = engine._do_export_gguf(output_path, force=force)
    except TypeError:
        result = engine._do_export_gguf(output_path)

    if result.get("ok") is True and result.get("files"):
        return {
            "ok": True,
            "status": result.get("status")
            or ("skipped" if result.get("skipped") else "exported"),
            "format": "gguf",
            "gguf_path": result.get("gguf_path") or gguf_dir,
            "files": list(result["files"]),
            "quants": quant_list,
            **{
                k: v
                for k, v in result.items()
                if k
                not in (
                    "ok",
                    "status",
                    "format",
                    "gguf_path",
                    "files",
                    "quants",
                    "error",
                )
            },
        }

    engine_error = result.get("error") or result.get("reason")
    if result.get("ok") is False or result.get("error"):
        return _export_failure(
            str(result.get("error") or engine_error or "GGUF failed"),
            format="gguf",
            quants=quant_list,
            gguf_path=gguf_dir,
            **{
                k: v
                for k, v in result.items()
                if k not in ("ok", "status", "error", "format")
            },
        )

    # Legacy / patched engine shapes — require verified artifacts on disk.
    verified = verify_gguf_artifacts(gguf_dir, quant_list)
    if not verified["ok"]:
        detail = verified.get("error") or "GGUF artifacts missing or empty"
        if engine_error and str(engine_error) not in detail:
            detail = f"{engine_error}. {detail}"
        return _export_failure(
            detail,
            format="gguf",
            gguf_path=gguf_dir,
            quants=quant_list,
            files=verified.get("files") or [],
            missing=verified.get("missing") or quant_list,
        )

    return {
        "ok": True,
        "status": "exported" if not result.get("skipped") else "skipped",
        "format": "gguf",
        "gguf_path": gguf_dir,
        "files": verified["files"],
        "quants": quant_list,
        **{
            k: v
            for k, v in result.items()
            if k
            not in (
                "ok",
                "status",
                "format",
                "gguf_path",
                "files",
                "quants",
                "error",
            )
        },
    }


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
        return _export_gguf(
            engine,
            output_path=output_path,
            quant_list=quant_list,
            force=force,
        )

    if fmt_norm == "abliterated":
        abl_dir = os.path.join(output_path, "abliterated")
        if os.path.isdir(abl_dir) and os.listdir(abl_dir) and not force:
            return {
                "ok": True,
                "status": "skipped",
                "format": "abliterated",
                "output_path": abl_dir,
                "message": (
                    "Abliterated model already exists. "
                    "Use force=true to overwrite."
                ),
            }
        result = engine._do_abliteration()
        if result.get("error"):
            # Never spread raw engine keys — may contain numpy arrays.
            return _export_failure(
                str(result["error"]),
                format="abliterated",
                reason=result.get("reason"),
            )
        if result.get("skipped"):
            return _export_failure(
                str(result.get("reason") or "abliteration skipped"),
                format="abliterated",
            )
        out_dir = str(result.get("output_dir") or abl_dir)
        layers = result.get("layers_modified") or result.get("layer_indices") or []
        return {
            "ok": True,
            "status": "exported",
            "format": "abliterated",
            "output_path": out_dir,
            "refusal_magnitude": float(result.get("refusal_magnitude") or 0.0),
            "layers_modified": [int(x) for x in layers],
            "strength": float(result.get("strength") or 1.0),
        }

    # gptq — fail fast when no backend; verify artifacts on success
    from finetune_studio.training.advanced_quant import (
        gptq_missing_backend_message,
        is_gptq_available,
        verify_gptq_artifacts,
    )

    if not is_gptq_available():
        return _export_failure(
            gptq_missing_backend_message(),
            format="gptq",
        )

    gptq_dir = os.path.join(output_path, "gptq")
    existing_gptq = verify_gptq_artifacts(gptq_dir)
    if existing_gptq["ok"] and not force:
        return {
            "ok": True,
            "status": "skipped",
            "format": "gptq",
            "output_path": gptq_dir,
            "files": existing_gptq.get("files") or [],
            "size_bytes": existing_gptq.get("size_bytes", 0),
            "message": "GPTQ already exists. Use force=true to overwrite.",
        }

    result = engine._do_export_gptq(output_path)
    if result.get("error"):
        return _export_failure(
            str(result["error"]),
            format="gptq",
            output_path=gptq_dir,
        )
    if result.get("skipped"):
        return _export_failure(
            str(result.get("reason", "gptq skipped")),
            format="gptq",
            output_path=gptq_dir,
        )

    verified = verify_gptq_artifacts(
        str(result.get("output_dir") or result.get("output_path") or gptq_dir)
    )
    if not verified["ok"]:
        return _export_failure(
            str(verified.get("error") or "GPTQ artifacts missing or empty"),
            format="gptq",
            output_path=gptq_dir,
            files=verified.get("files") or [],
        )
    return {
        "ok": True,
        "status": "exported",
        "format": "gptq",
        "output_path": verified.get("output_dir") or gptq_dir,
        "files": verified.get("files") or [],
        "size_bytes": verified.get("size_bytes") or result.get("size_bytes"),
        "size_human": result.get("size_human"),
        "bits": result.get("bits"),
        "group_size": result.get("group_size"),
    }
