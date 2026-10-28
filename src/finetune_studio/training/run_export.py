"""Post-training export: ensure merged weights, then convert to deployable formats.

Supports raw adapter-only runs by merging onto a compatible 16-bit base at
export time (``base_model`` override or the run's stored base). AWQ and
GPTQ were removed (unmaintained); supported formats are ``gguf``,
``abliterated``, and ``merged`` (safetensors only).

GGUF exports only report success when the requested ``.gguf`` artifact(s)
exist on disk and are non-empty — never on HTTP 200 / empty dirs alone.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from finetune_studio.training import export_job
from finetune_studio.training.gguf_convert import (  # noqa: F401 — re-export
    GGUF_CONVERTER_MISSING_MSG,
    LLAMA_IMATRIX_MISSING_MSG,
    LLAMA_QUANTIZE_MISSING_MSG,
    convert_merged_to_gguf,
    find_gguf_convert_script,
    find_llama_imatrix,
    find_llama_quantize,
    gguf_filename_matches_quant,
    imatrix_applies,
    llama_cpp_search_paths,
    normalize_gguf_quant,
    verify_gguf_artifacts,
)

SUPPORTED_EXPORT_FORMATS: frozenset[str] = frozenset(
    {"gguf", "abliterated", "merged", "heretic"}
)

DEFAULT_GGUF_QUANTS: list[str] = ["f16", "q8_0", "q4_k_m", "q5_k_m"]

# An importance-matrix export lands beside the plain one, never over it (same quant names, different weights).
IMATRIX_SUBDIR = "gguf-imatrix"
IMATRIX_CALIBRATION_CHARS = 300_000   # ~75k tokens: more than the 128 chunks x 512 tokens llama-imatrix measures


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

    export_job.report(export_job.PHASE_MERGING, "merging adapter into base weights")
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


def imatrix_calibration_text(data_path: str, *, max_chars: int = IMATRIX_CALIBRATION_CHARS) -> str:
    """The run's training text (every turn of every row, in file order) for measuring an importance matrix.

    Reads JSONL rows in the shapes training accepts (``messages`` / ``conversations`` turns, or ``text`` /
    ``prompt``+``completion``) and stops at ``max_chars``. Empty when the file is missing or holds no text.
    """
    import json

    if not data_path or not os.path.isfile(data_path):
        return ""
    parts: list[str] = []
    total = 0
    with open(data_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            turns = row.get("messages") or row.get("conversations") or []
            texts = [str(t.get("content") or t.get("value") or "") for t in turns if isinstance(t, dict)]
            if not texts:
                texts = [str(row.get(k) or "") for k in ("text", "prompt", "completion", "instruction", "output")]
            block = "\n".join(t.strip() for t in texts if t.strip())
            if not block:
                continue
            parts.append(block)
            total += len(block) + 2
            if total >= max_chars:
                break
    return "\n\n".join(parts)[:max_chars]


def _export_gguf(
    engine: Any,
    *,
    output_path: str,
    quant_list: list[str],
    force: bool,
    subdir: str = "gguf",
    calibration_text: str = "",
) -> dict[str, Any]:
    """Run GGUF conversion and only succeed when artifacts verify non-empty."""
    gguf_dir = os.path.join(output_path, subdir)
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
    if calibration_text or subdir != "gguf":
        result = engine._do_export_gguf(output_path, force=force, subdir=subdir, calibration_text=calibration_text)
    else:
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
        # ``result`` usually carries gguf_path/quants itself: merge, never
        # pass them twice (that raised TypeError -> HTTP 500 on every real
        # converter failure).
        return _export_failure(
            str(result.get("error") or engine_error or "GGUF failed"),
            format="gguf",
            **{
                "quants": quant_list,
                "gguf_path": gguf_dir,
                **{
                    k: v
                    for k, v in result.items()
                    if k not in ("ok", "status", "error", "format")
                },
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


def _export_heretic(
    *,
    merged_path: str,
    output_path: str,
    force: bool,
    n_trials: int | None = None,
) -> dict[str, Any]:
    """Remove refusals with the optional Heretic tool, into ``<output>/heretic/``.

    Heretic is a separate program (AGPL-3.0) that searches abliteration
    parameters with Optuna; this only drives it and verifies what it wrote.
    A run that writes weights without ``config.json``/a tokenizer is a failure
    here, not an export: such a directory cannot be loaded by Chat, Testing or
    any other app.
    """
    from finetune_studio.training import heretic_tool
    from finetune_studio.training.proc_runner import (
        SubprocessCancelled,
        SubprocessFailed,
        SubprocessTimeout,
    )

    out_dir = os.path.join(output_path, "heretic")
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not force:
        return {
            "ok": True,
            "status": "skipped",
            "format": "heretic",
            "output_path": out_dir,
            "message": "A Heretic model already exists here. Use force=true to overwrite.",
        }
    if not merged_path or not os.path.isdir(merged_path):
        # Heretic ablates a full model; a bare adapter is not one. Merging happens
        # before this call, so an absent merged/ is a real failure, not a skip.
        return _export_failure(
            "no merged model to abliterate; export Merged (or retrain with merge on save) first",
            format="heretic",
        )

    status = heretic_tool.probe()
    if not status.usable:
        return _export_failure(
            status.hint or "Heretic is not available",
            format="heretic",
            install_command="fts heretic install",
        )

    export_job.report(export_job.PHASE_ABLITERATING, "Heretic: searching abliteration parameters")
    ctx = export_job.current()
    run = heretic_tool.AbliterationRun(
        model_path=merged_path,
        output_dir=out_dir,
        n_trials=int(n_trials or 40),
        checkpoint_dir=os.path.join(out_dir, ".study"),
    )
    try:
        written = heretic_tool.run_abliteration(
            run,
            cancelled=ctx.cancelled if ctx else None,
            on_tick=ctx.tick if ctx else None,
            on_start=ctx.register_proc if ctx else None,
        )
    except SubprocessCancelled as e:
        raise export_job.ExportCancelled(str(e)) from e
    except SubprocessTimeout as e:
        return _export_failure(f"Heretic did not finish in time: {e}", format="heretic")
    except SubprocessFailed as e:
        return _export_failure(f"Heretic failed: {e.tail or e}", format="heretic")
    except heretic_tool.HereticError as e:
        return _export_failure(str(e), format="heretic")

    return {
        "ok": True,
        "status": "exported",
        "format": "heretic",
        "output_path": written["output_path"],
        "size_bytes": written["size_bytes"],
        "version": status.version,
        "ref": status.ref,
        "files": written["files"],
    }


def validate_export_request(
    run: dict[str, Any],
    *,
    fmt: str = "gguf",
    quants: list[str] | None = None,
    force: bool = False,
    imatrix: bool = False,
) -> dict[str, Any] | None:
    """Cheap pre-flight: a failure payload for a request that cannot succeed.

    Pure checks (format name, output dir, converter present) so the WebUI can
    answer 400 immediately instead of queueing a job that fails minutes later
    after a multi-GB merge. Returns ``None`` when the request may proceed.
    """
    fmt_norm = (fmt or "gguf").strip().lower()
    if fmt_norm == "awq":
        return _export_failure(
            "AWQ export was removed (autoawq unmaintained). "
            "Use format=gguf or merged instead.",
            format="awq",
        )
    if fmt_norm == "gptq":
        return _export_failure(
            "GPTQ export was removed (gptqmodel's fragile upstream "
            "packaging metadata made it unreliable to install). "
            "Use format=gguf or merged instead.",
            format="gptq",
        )
    if fmt_norm not in SUPPORTED_EXPORT_FORMATS:
        return _export_failure(
            f"unknown format: {fmt_norm}",
            format=fmt_norm,
            supported=sorted(SUPPORTED_EXPORT_FORMATS),
        )
    if not (run.get("output_path") or "").strip():
        return _export_failure("run has no output_path", format=fmt_norm)
    if fmt_norm == "heretic":
        # Heretic is an optional external tool: refuse here, with the reason and the
        # fix, instead of queueing a job that dies after a multi-GB merge.
        from finetune_studio.training import heretic_tool

        status = heretic_tool.probe()
        if not status.usable:
            return _export_failure(
                status.hint or "Heretic is not available", format="heretic",
                install_command="fts heretic install",
            )
    if imatrix:
        refused = _imatrix_refusal(run, fmt_norm, list(quants or DEFAULT_GGUF_QUANTS))
        if refused is not None:
            return refused
    if (
        fmt_norm == "gguf"
        and find_gguf_convert_script() is None
        and os.environ.get("FTS_SKIP_EXPORT") != "1"
    ):
        # Without a converter the only successful outcome is "already
        # exported, nothing to do" (see ``_export_gguf``).
        gguf_dir = os.path.join((run.get("output_path") or "").strip(), "gguf")
        existing = verify_gguf_artifacts(gguf_dir, list(quants or DEFAULT_GGUF_QUANTS))
        if force or not existing["ok"]:
            return _export_failure(GGUF_CONVERTER_MISSING_MSG, format="gguf")
    return None


def _imatrix_refusal(run: dict[str, Any], fmt_norm: str, quants: list[str]) -> dict[str, Any] | None:
    """Why an importance-matrix export cannot run, or None."""
    if fmt_norm != "gguf":
        return _export_failure("the importance matrix only applies to a GGUF export", format=fmt_norm)
    if not imatrix_applies(quants):
        return _export_failure(
            "the importance matrix only changes llama-quantize types (Q4_K_M, Q5_K_M, Q6_K, IQ…); "
            "f16, bf16 and Q8_0 are written without it, so pick at least one of those",
            format="gguf", quants=quants)
    if os.environ.get("FTS_SKIP_EXPORT") != "1" and find_llama_imatrix() is None:
        return _export_failure(LLAMA_IMATRIX_MISSING_MSG, format="gguf")
    if not imatrix_calibration_text(str(run.get("data_path") or ""), max_chars=1):
        return _export_failure(
            "this run's training data file is missing or empty, so there is no text to measure the importance "
            "matrix on; export without it", format="gguf")
    return None


def export_trained_run(
    run: dict[str, Any],
    *,
    fmt: str = "gguf",
    quants: list[str] | None = None,
    force: bool = False,
    base_model: str | None = None,
    imatrix: bool = False,
) -> dict[str, Any]:
    """Merge if needed, then export ``run`` to ``fmt``.

    ``imatrix`` (GGUF only) measures an importance matrix on the run's training text and quantizes with it, into
    ``<output>/gguf-imatrix/``. Returns a dict with ``ok`` / ``status`` / ``error`` plus format fields.
    Failures always set ``ok=False`` and ``status="failed"``.
    """
    fmt_norm = (fmt or "gguf").strip().lower()
    refused = validate_export_request(
        run, fmt=fmt_norm, quants=quants, force=force, imatrix=imatrix,
    )
    if refused is not None:
        return refused
    output_path = (run.get("output_path") or "").strip()

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

    if fmt_norm == "heretic":
        return _export_heretic(
            merged_path=str(merge_info.get("merged_path") or ""),
            output_path=output_path,
            force=force,
        )

    from finetune_studio.training.engine import TrainingConfig, TrainingEngine

    quant_list = list(quants) if quants else list(DEFAULT_GGUF_QUANTS)
    cfg = TrainingConfig(output_dir=output_path, gguf_quants=quant_list)
    engine = TrainingEngine()
    engine.config = cfg

    if fmt_norm == "gguf":
        if imatrix:
            return _export_gguf(
                engine, output_path=output_path, quant_list=quant_list, force=force, subdir=IMATRIX_SUBDIR,
                calibration_text=imatrix_calibration_text(str(run.get("data_path") or "")),
            )
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
        export_job.report(
            export_job.PHASE_ABLITERATING, "measuring + removing refusal direction",
        )
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

    return _export_failure(f"unknown format: {fmt_norm}", format=fmt_norm)
