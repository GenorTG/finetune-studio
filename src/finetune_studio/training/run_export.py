"""Post-training export: ensure merged weights, then convert to deployable formats.

Supports raw adapter-only runs by merging onto a compatible 16-bit base at
export time (``base_model`` override or the run's stored base). AWQ was
removed (unmaintained); supported formats are ``gguf``, ``gptq``,
``abliterated``, and ``merged`` (safetensors only).

GGUF exports only report success when the requested ``.gguf`` artifact(s)
exist on disk and are non-empty — never on HTTP 200 / empty dirs alone.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SUPPORTED_EXPORT_FORMATS: frozenset[str] = frozenset(
    {"gguf", "gptq", "abliterated", "merged"}
)

DEFAULT_GGUF_QUANTS: list[str] = ["f16", "q8_0", "q4_k_m", "q5_k_m"]

GGUF_CONVERTER_MISSING_MSG: str = (
    "convert_hf_to_gguf.py not found. Install llama.cpp on this host: "
    "git clone https://github.com/ggerganov/llama.cpp && "
    "pip install -r llama.cpp/requirements/"
    "requirements-convert_hf_to_gguf.txt. "
    "Project-local .llama.cpp/ is also searched. "
    "Or choose format=merged until the converter is installed."
)


def _project_root() -> str:
    """Repo root containing ``.llama.cpp/`` (``src/finetune_studio/training/``)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def llama_cpp_search_paths() -> list[str]:
    """Ordered roots where ``convert_hf_to_gguf.py`` may live."""
    return [
        os.path.join(_project_root(), ".llama.cpp"),
        os.path.expanduser("~/llama.cpp"),
        "/opt/llama.cpp",
        "/usr/local/llama.cpp",
    ]


def find_gguf_convert_script() -> str | None:
    """Locate llama.cpp ``convert_hf_to_gguf.py``, or None if missing."""
    for base in llama_cpp_search_paths():
        candidate = os.path.join(base, "convert_hf_to_gguf.py")
        if os.path.isfile(candidate):
            return candidate
    # Legacy filenames / PATH-adjacent installs
    for candidate in (
        os.path.expanduser("~/llama.cpp/convert.py"),
        os.path.expanduser("~/llama.cpp/convert-hf-to-gguf.py"),
        "/usr/local/bin/convert-hf-to-gguf.py",
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def normalize_gguf_quant(quant: str) -> str:
    """Normalize a quant label for filename matching (``Q4_K_M`` → ``q4_k_m``)."""
    return (quant or "").strip().lower().replace("-", "_").replace(".", "_")


def gguf_filename_matches_quant(filename: str, quant: str) -> bool:
    """True when ``filename`` is a ``.gguf`` whose stem ends with the quant."""
    name = os.path.basename(filename).lower()
    if not name.endswith(".gguf"):
        return False
    stem = name[:-5]
    nq = normalize_gguf_quant(quant)
    if not nq:
        return False
    if stem == nq or stem == f"model-{nq}":
        return True
    return stem.endswith((f"-{nq}", f"_{nq}"))


def verify_gguf_artifacts(
    gguf_dir: str,
    quants: list[str] | None = None,
) -> dict[str, Any]:
    """Require non-empty ``.gguf`` files for ``quants`` (or any if unset).

    Returns ``ok``, ``files``, ``missing``, ``error``. Never raises.
    """
    wanted = [normalize_gguf_quant(q) for q in (quants or []) if str(q).strip()]
    if not os.path.isdir(gguf_dir):
        return {
            "ok": False,
            "files": [],
            "missing": wanted,
            "error": f"GGUF directory missing: {gguf_dir}",
        }
    try:
        names = os.listdir(gguf_dir)
    except OSError as e:
        return {
            "ok": False,
            "files": [],
            "missing": wanted,
            "error": f"Cannot read GGUF directory {gguf_dir}: {e}",
        }

    nonempty: list[str] = []
    for name in names:
        if not name.lower().endswith(".gguf"):
            continue
        path = os.path.join(gguf_dir, name)
        try:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                nonempty.append(path)
        except OSError:
            continue

    if not nonempty:
        return {
            "ok": False,
            "files": [],
            "missing": wanted,
            "error": (
                "No non-empty .gguf files found after export. "
                "Install llama.cpp (convert_hf_to_gguf.py + llama-quantize) "
                "and retry, or choose format=merged."
            ),
        }

    if not wanted:
        return {
            "ok": True,
            "files": nonempty,
            "missing": [],
            "error": None,
        }

    matched: list[str] = []
    missing: list[str] = []
    for nq in wanted:
        hit = next(
            (p for p in nonempty if gguf_filename_matches_quant(p, nq)),
            None,
        )
        if hit is None:
            missing.append(nq)
        else:
            matched.append(hit)

    if missing:
        return {
            "ok": False,
            "files": matched,
            "missing": missing,
            "error": (
                "Missing non-empty GGUF artifact(s) for: "
                + ", ".join(missing)
                + ". Conversion may have failed or llama.cpp tools are incomplete."
            ),
        }
    return {"ok": True, "files": matched, "missing": [], "error": None}


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

    if find_gguf_convert_script() is None:
        return _export_failure(
            GGUF_CONVERTER_MISSING_MSG,
            format="gguf",
            gguf_path=gguf_dir,
            quants=quant_list,
            missing=quant_list,
        )

    result = engine._do_export_gguf(output_path)
    # Engine historically returned skipped/reason without error — treat as fail
    # unless non-empty artifacts for the requested quants are on disk.
    engine_error = result.get("error") or result.get("reason")
    if result.get("error"):
        return _export_failure(
            str(result["error"]),
            format="gguf",
            quants=quant_list,
            **{k: v for k, v in result.items() if k not in ("ok", "status")},
        )

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
            **{
                k: v
                for k, v in result.items()
                if k
                not in (
                    "ok",
                    "status",
                    "error",
                    "gguf_path",
                    "quants",
                    "files",
                    "missing",
                )
            },
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
