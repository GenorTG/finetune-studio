"""Detect which export converters are available on this host.

Used by the Export page to disable GGUF when tooling is missing, so the
UI never advertises a format that can only fail (or historically reported
fake success). Always-available formats: ``merged``, ``abliterated``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from finetune_studio.training.run_export import (
    GGUF_CONVERTER_MISSING_MSG,
    LLAMA_IMATRIX_MISSING_MSG,
    LLAMA_QUANTIZE_MISSING_MSG,
    find_gguf_convert_script,
    find_llama_imatrix,
    find_llama_quantize,
)


@dataclass(frozen=True)
class ExportCapabilities:
    """Host-local converter availability for the Export UI."""

    gguf: bool
    gguf_script: str | None
    gguf_hint: str
    imatrix: bool = False          # llama-imatrix + llama-quantize present: the "Importance matrix" GGUF option works
    imatrix_hint: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable view for templates / API."""
        return asdict(self)


def probe_export_capabilities() -> ExportCapabilities:
    """Probe the filesystem for the GGUF converter and the importance-matrix tools."""
    script = find_gguf_convert_script()
    gguf_ok = script is not None
    imatrix_hint = ("" if find_llama_imatrix() else LLAMA_IMATRIX_MISSING_MSG) or (
        "" if find_llama_quantize() else LLAMA_QUANTIZE_MISSING_MSG)
    return ExportCapabilities(
        gguf=gguf_ok,
        gguf_script=script,
        gguf_hint="" if gguf_ok else GGUF_CONVERTER_MISSING_MSG,
        imatrix=gguf_ok and not imatrix_hint,
        imatrix_hint=imatrix_hint if gguf_ok else GGUF_CONVERTER_MISSING_MSG,
    )
