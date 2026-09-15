"""JSON-safe typed payloads for sync/async training-run exports.

Abliteration and other GPU paths may attach numpy arrays / tensors to their
raw result dicts. FastAPI's ``jsonable_encoder`` then raises TypeError
(dict/vars) and the client sees HTTP 500 after a successful merge. This
module is the single place that turns those raw dicts into a serializable
``ExportResult``.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _is_json_primitive(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _to_jsonable(value: Any) -> Any:
    """Coerce one value to a JSON-serializable form, or drop it."""
    if _is_json_primitive(value):
        return value
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            coerced = _to_jsonable(item)
            if coerced is _DROP:
                continue
            out.append(coerced)
        return out
    if isinstance(value, dict):
        return {
            str(k): v
            for k, v in ((str(k), _to_jsonable(v)) for k, v in value.items())
            if v is not _DROP
        }
    # numpy / torch scalars
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError, AttributeError):
            scalar = _DROP
        else:
            if _is_json_primitive(scalar):
                return scalar
    # numpy arrays / tensors — never embed full vectors in API responses
    if hasattr(value, "tolist") and hasattr(value, "shape"):
        return _DROP
    if hasattr(value, "tolist") and not hasattr(value, "shape"):
        try:
            return _to_jsonable(value.tolist())
        except (TypeError, ValueError, AttributeError):
            return _DROP
    return _DROP


_DROP = object()


def sanitize_export_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Strip non-JSON values and normalize path field names.

    Maps ``output_dir`` → ``output_path`` when the latter is absent so the
    Export UI can display the artifact path from a single key.
    """
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        coerced = _to_jsonable(value)
        if coerced is _DROP:
            continue
        cleaned[str(key)] = coerced

    if not cleaned.get("output_path") and cleaned.get("output_dir"):
        cleaned["output_path"] = cleaned["output_dir"]
    return cleaned


class ExportResult(BaseModel):
    """Typed export API response — always JSON-encodable."""

    model_config = ConfigDict(extra="ignore")

    ok: bool
    status: str
    format: str | None = None
    error: str | None = None
    message: str | None = None
    export_id: str | None = None
    output_path: str | None = None
    merged_path: str | None = None
    gguf_path: str | None = None
    path: str | None = None
    files: list[str] = Field(default_factory=list)
    quants: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    quant: str | None = None
    size_bytes: int | None = None
    size_human: str | None = None
    refusal_magnitude: float | None = None
    layers_modified: list[int] = Field(default_factory=list)
    strength: float | None = None
    supported: list[str] = Field(default_factory=list)
    skipped: bool | None = None
    reason: str | None = None

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> ExportResult:
        """Build from a raw export/engine dict (numpy-safe)."""
        data = sanitize_export_dict(dict(raw))
        # Normalize failure shape
        if data.get("error") or data.get("ok") is False:
            data["ok"] = False
            data.setdefault("status", "failed")
        else:
            data.setdefault("ok", True)
            data.setdefault("status", "exported")
        # Prefer explicit paths for display
        if not data.get("output_path"):
            data["output_path"] = (
                data.get("merged_path")
                or data.get("gguf_path")
                or data.get("path")
            )
        # Coerce list fields that may arrive as other iterables
        for list_key in ("files", "quants", "missing", "layers_modified", "supported"):
            val = data.get(list_key)
            if val is None:
                data[list_key] = []
            elif not isinstance(val, list):
                data[list_key] = list(val)
        return cls.model_validate(data)

    def artifact_path(self) -> str:
        """Best path for UI / DB registration."""
        return (
            (self.output_path or "")
            or (self.merged_path or "")
            or (self.gguf_path or "")
            or (self.path or "")
            or (self.files[0] if self.files else "")
        )


def dir_size_bytes(path: str) -> int:
    """Total bytes under ``path`` (files only); 0 if missing."""
    if not path or not os.path.isdir(path):
        if path and os.path.isfile(path):
            try:
                return os.path.getsize(path)
            except OSError:
                return 0
        return 0
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
    except OSError:
        return 0
    return total


def human_size(n: int) -> str:
    """Human-readable size string for export rows."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} PB"
