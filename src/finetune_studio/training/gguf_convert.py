"""llama.cpp GGUF conversion — discovery + HF→GGUF + artifact checks.

Single source of truth for convert_hf_to_gguf.py / llama-quantize discovery
and conversion. Used by sync export (``run_export`` / ``TrainingEngine``)
and the legacy async export worker.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from typing import Any

log = logging.getLogger(__name__)

GGUF_CONVERTER_MISSING_MSG: str = (
    "convert_hf_to_gguf.py not found. Install llama.cpp on this host: "
    "git clone https://github.com/ggerganov/llama.cpp && "
    "pip install -r llama.cpp/requirements/"
    "requirements-convert_hf_to_gguf.txt. "
    "Project-local .llama.cpp/ is also searched. "
    "Or choose format=merged until the converter is installed."
)

LLAMA_QUANTIZE_MISSING_MSG: str = (
    "llama-quantize binary not found (needed for K-quants / IQ quants). "
    "Build llama.cpp: cmake -B build && cmake --build build --config Release. "
    "Searched PATH, .llama.cpp/build/bin/, ~/llama.cpp/build/bin/."
)

# Quants convertible in one step via convert_hf_to_gguf.py --outtype.
# Everything else: HF → f16 GGUF → llama-quantize.
_SINGLE_STEP_OUTTYPES: dict[str, str] = {
    "f16": "f16",
    "bf16": "bf16",
    "f32": "f32",
    "q8_0": "q8_0",
}


def _project_root() -> str:
    """Repo root (``src/finetune_studio/training/`` → three levels up)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def llama_cpp_search_paths() -> list[str]:
    """Ordered roots where ``convert_hf_to_gguf.py`` / build bins may live."""
    env = (os.environ.get("LLAMA_CPP_DIR") or "").strip()
    paths: list[str] = []
    if env:
        paths.append(os.path.expanduser(env))
    paths.extend(
        [
            os.path.join(_project_root(), ".llama.cpp"),
            os.path.expanduser("~/llama.cpp"),
            "/opt/llama.cpp",
            "/usr/local/llama.cpp",
        ]
    )
    # Dedupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def find_gguf_convert_script() -> str | None:
    """Locate llama.cpp ``convert_hf_to_gguf.py``, or None if missing."""
    for base in llama_cpp_search_paths():
        candidate = os.path.join(base, "convert_hf_to_gguf.py")
        if os.path.isfile(candidate):
            return candidate
    for candidate in (
        os.path.expanduser("~/llama.cpp/convert.py"),
        os.path.expanduser("~/llama.cpp/convert-hf-to-gguf.py"),
        "/usr/local/bin/convert-hf-to-gguf.py",
        shutil.which("convert-hf-to-gguf.py") or "",
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def find_llama_quantize() -> str | None:
    """Locate ``llama-quantize`` executable (PATH + common build dirs)."""
    which = shutil.which("llama-quantize")
    if which and os.path.isfile(which) and os.access(which, os.X_OK):
        return which
    for base in llama_cpp_search_paths():
        for rel in (
            os.path.join("build", "bin", "llama-quantize"),
            os.path.join("build-RPC", "bin", "llama-quantize"),
            "llama-quantize",
            "quantize",
        ):
            candidate = os.path.join(base, rel)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    # Legacy bare name next to convert script's parent build
    script = find_gguf_convert_script()
    if script:
        sibling = os.path.normpath(
            os.path.join(os.path.dirname(script), "build", "bin", "llama-quantize")
        )
        if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
            return sibling
    return None


def normalize_gguf_quant(quant: str) -> str:
    """Normalize a quant label for filenames (``Q4_K_M`` → ``q4_k_m``)."""
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


def _run_cmd(cmd: list[str], *, timeout: int = 3600) -> None:
    """Run a subprocess; raise RuntimeError with stderr tail on failure."""
    log.info("gguf convert: %s", " ".join(cmd))
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False,
    )
    if result.returncode != 0:
        err_tail = (result.stderr or result.stdout or "")[-1000:]
        raise RuntimeError(
            f"{cmd[0]} failed (rc={result.returncode}): {err_tail}"
        )


def _needs_quantize_bin(quants: list[str]) -> bool:
    return any(
        normalize_gguf_quant(q) not in _SINGLE_STEP_OUTTYPES for q in quants
    )


def convert_merged_to_gguf(
    merged_dir: str,
    gguf_dir: str,
    quants: list[str],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Convert a merged HF directory to GGUF files for each quant.

    Success only when every requested quant has a non-empty ``.gguf`` on disk.
    ``FTS_SKIP_EXPORT=1`` writes 1-byte markers (tests) without calling llama.cpp.

    Returns ``ok``, ``gguf_path``, ``files``, ``exported``, ``error``, …
    """
    quant_list = [normalize_gguf_quant(q) for q in quants if str(q).strip()]
    if not quant_list:
        return {
            "ok": False,
            "gguf_path": gguf_dir,
            "files": [],
            "error": "no quants requested",
            "status": "failed",
        }

    existing = verify_gguf_artifacts(gguf_dir, quant_list)
    if existing["ok"] and not force:
        return {
            "ok": True,
            "status": "skipped",
            "gguf_path": gguf_dir,
            "files": existing["files"],
            "quants": quant_list,
            "skipped": True,
            "message": "GGUF already exists. Use force=true to overwrite.",
        }

    os.makedirs(gguf_dir, exist_ok=True)

    # Test / CI short-circuit — non-empty markers so verify passes.
    # Runs before the merged-dir check so worker unit tests can use empty fixtures.
    if os.environ.get("FTS_SKIP_EXPORT") == "1":
        files: list[str] = []
        exported: dict[str, Any] = {}
        for nq in quant_list:
            out_path = os.path.join(gguf_dir, f"model-{nq}.gguf")
            with open(out_path, "wb") as f:
                f.write(b"\x00")
            files.append(out_path)
            exported[nq] = {"path": out_path, "size": 1}
        return {
            "ok": True,
            "status": "exported",
            "gguf_path": gguf_dir,
            "files": files,
            "quants": quant_list,
            "exported": exported,
            "skipped": False,
        }

    if not os.path.isdir(merged_dir) or not os.listdir(merged_dir):
        return {
            "ok": False,
            "gguf_path": gguf_dir,
            "files": [],
            "skipped": True,
            "reason": "no merged model to convert",
            "error": "no merged model to convert",
            "status": "failed",
        }

    convert_script = find_gguf_convert_script()
    if not convert_script:
        return {
            "ok": False,
            "status": "failed",
            "gguf_path": gguf_dir,
            "files": [],
            "quants": quant_list,
            "missing": quant_list,
            "error": GGUF_CONVERTER_MISSING_MSG,
        }

    quant_bin: str | None = None
    if _needs_quantize_bin(quant_list):
        quant_bin = find_llama_quantize()
        if not quant_bin:
            return {
                "ok": False,
                "status": "failed",
                "gguf_path": gguf_dir,
                "files": [],
                "quants": quant_list,
                "error": LLAMA_QUANTIZE_MISSING_MSG,
            }

    python = sys.executable
    exported_map: dict[str, Any] = {}
    intermediate_fp16 = ""

    try:
        # Shared fp16 intermediate when any multi-step quant is requested.
        need_fp16 = any(q not in _SINGLE_STEP_OUTTYPES for q in quant_list)
        fp16_path = os.path.join(gguf_dir, "model-f16.gguf")

        if "f16" in quant_list or need_fp16:
            if force or not (
                os.path.isfile(fp16_path) and os.path.getsize(fp16_path) > 0
            ):
                _run_cmd(
                    [
                        python,
                        convert_script,
                        merged_dir,
                        "--outfile",
                        fp16_path,
                        "--outtype",
                        "f16",
                    ]
                )
            if "f16" in quant_list:
                exported_map["f16"] = {
                    "path": fp16_path,
                    "size": os.path.getsize(fp16_path),
                }
            if need_fp16:
                intermediate_fp16 = fp16_path

        for nq in quant_list:
            if nq == "f16":
                continue
            out_path = os.path.join(gguf_dir, f"model-{nq}.gguf")
            if nq in _SINGLE_STEP_OUTTYPES:
                _run_cmd(
                    [
                        python,
                        convert_script,
                        merged_dir,
                        "--outfile",
                        out_path,
                        "--outtype",
                        _SINGLE_STEP_OUTTYPES[nq],
                    ]
                )
            else:
                assert quant_bin is not None
                # llama-quantize wants uppercase-ish type labels (Q4_K_M).
                quant_arg = nq.upper()
                _run_cmd([quant_bin, fp16_path, out_path, quant_arg])
            if not os.path.isfile(out_path) or os.path.getsize(out_path) <= 0:
                raise RuntimeError(
                    f"output GGUF missing or empty after conversion: {out_path}"
                )
            exported_map[nq] = {
                "path": out_path,
                "size": os.path.getsize(out_path),
            }
    except Exception as e:
        log.exception("GGUF conversion failed")
        return {
            "ok": False,
            "status": "failed",
            "gguf_path": gguf_dir,
            "files": [],
            "quants": quant_list,
            "error": str(e),
            "intermediate_path": intermediate_fp16 or None,
        }

    verified = verify_gguf_artifacts(gguf_dir, quant_list)
    if not verified["ok"]:
        return {
            "ok": False,
            "status": "failed",
            "gguf_path": gguf_dir,
            "files": verified.get("files") or [],
            "missing": verified.get("missing") or quant_list,
            "quants": quant_list,
            "exported": exported_map,
            "error": verified.get("error") or "GGUF artifacts missing or empty",
            "intermediate_path": intermediate_fp16 or None,
        }

    size = sum(
        int(v.get("size") or 0)
        for v in exported_map.values()
        if isinstance(v, dict)
    )
    return {
        "ok": True,
        "status": "exported",
        "gguf_path": gguf_dir,
        "files": verified["files"],
        "quants": quant_list,
        "exported": exported_map,
        "skipped": False,
        "size_bytes": size,
        "intermediate_path": intermediate_fp16 or None,
    }
