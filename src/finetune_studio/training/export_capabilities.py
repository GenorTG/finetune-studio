"""Detect which export converters are available on this host.

Used by the Export page to disable GGUF/GPTQ when tooling is missing, so the
UI never advertises a format that can only fail (or historically reported
fake success). Always-available formats: ``merged``, ``abliterated``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from finetune_studio.training.advanced_quant import (
    gptq_dependency_hints,
    gptq_missing_backend_message,
    is_gptq_available,
    is_optimum_available,
)
from finetune_studio.training.run_export import (
    GGUF_CONVERTER_MISSING_MSG,
    find_gguf_convert_script,
)

GPTQ_CONVERTER_MISSING_MSG = gptq_missing_backend_message()


@dataclass(frozen=True)
class ExportCapabilities:
    """Host-local converter availability for the Export UI."""

    gguf: bool
    gptq: bool
    gguf_script: str | None
    gguf_hint: str
    gptq_hint: str
    optimum: bool = False
    gptq_inference_hf: bool = False

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable view for templates / API."""
        return asdict(self)


def probe_export_capabilities() -> ExportCapabilities:
    """Probe the filesystem / imports for GGUF and GPTQ converters."""
    script = find_gguf_convert_script()
    gguf_ok = script is not None
    gptq_ok = is_gptq_available()
    deps = gptq_dependency_hints()
    optimum_ok = bool(deps["optimum"])
    if not gptq_ok:
        gptq_hint = GPTQ_CONVERTER_MISSING_MSG
    elif not optimum_ok:
        # Export can proceed; surface inference-gap so the UI is not silent.
        gptq_hint = str(deps["hint"])
    else:
        gptq_hint = ""
    return ExportCapabilities(
        gguf=gguf_ok,
        gptq=gptq_ok,
        gguf_script=script,
        gguf_hint="" if gguf_ok else GGUF_CONVERTER_MISSING_MSG,
        gptq_hint=gptq_hint,
        optimum=optimum_ok if gptq_ok else is_optimum_available(),
        gptq_inference_hf=bool(deps["gptq_inference_hf"]),
    )
