"""The Vision reader seat: the model that reads pictures in documents (scans, chart and table images) during mining.

Its own setting, separate from the helper seat (Genor 2026-10-14): document images leave the machine only when an API vision
model is picked here on purpose.

- ``off`` (default): figures keep their OCR text, marked "read by OCR, not checked".
- ``local``: a GGUF with an mmproj next to it, loaded in the ONE inference worker. If it was not loaded before a batch, it
  is unloaded after. A local helper that is a DIFFERENT GGUF would be evicted by it in the middle of mining, so that
  combination is refused with a reason instead.
- ``api``: an API connection that accepts images (``provider_id``; older installs: the ``api-vision`` row).
- ``assistant``: whatever the main assistant is (``models.assistant``), resolved when a batch is queued; a local
  assistant needs an mmproj next to its GGUF.

Settings -> AI assistant -> Jobs -> "Reading pictures" picks the seat; the job's extra instructions go in front of every
reading.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

VISION_SEAT_KEY = "vision_reader"
API_VISION_PROVIDER_ID = "api-vision"
MODES = ("off", "local", "api", "assistant")
MAX_TOKENS = 4096
MAX_SIDE_PX = 1536        # bounds a picture's image tokens (a 200 dpi page is ~1700x2200 px)
LOCAL_N_CTX = 16384       # a reading needs the picture + prompt + answer, not the model's 262k window

READ_PROMPT = """This picture comes from a page of the document "{title}" ({where}). Transcribe it so that a reader who \
cannot see it loses nothing:
- every piece of text, word for word, in reading order (titles, labels, notes, handwriting), one line per line, field or \
list item of the page (not one long paragraph);
- a table as rows "col | col | col" with its header row first, every cell exactly as written;
- a chart as: its title and units, then one row per data point "category | series | value" with every value as printed on \
the chart (read printed labels; if a value is not printed, give your reading and mark it "(estimated)");
- a photo or drawing: one sentence on what it shows, then any text in it.
Never add facts that are not visible. Output only the transcription."""


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CHANNEL = re.compile(r"^.*<channel\|>", re.DOTALL)   # Gemma 4: '<|channel>thought ... <channel|>' before the answer
# Markdown escapes: a blank form field came back as "Closed by: \_\_\_\_ (Kowarzik)" and was mined with the backslashes.
_MD_ESCAPE = re.compile(r"\\([_*#`|~\[\]])")


def clean(text: str) -> str:
    """The transcription without reasoning blocks (a local Gemma 4 prefixed every reading with its thought channel) and
    without markdown escapes."""
    text = _CHANNEL.sub("", _THINK.sub("", text or ""))
    return _MD_ESCAPE.sub(r"\1", text).lstrip("> \n").strip()


class VisionError(RuntimeError):
    pass


@dataclass
class Seat:
    mode: str = "off"
    local_model: str = ""          # GGUF path (local mode)
    provider_id: str = ""          # API connection (api mode); '' = the legacy ``api-vision`` row

    def as_json(self) -> str:
        return json.dumps({"mode": self.mode, "local_model": self.local_model, "provider_id": self.provider_id})

    def as_dict(self) -> dict[str, str]:
        return {"mode": self.mode, "local_model": self.local_model, "provider_id": self.provider_id}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Seat:
        mode = raw.get("mode") if raw.get("mode") in MODES else "off"
        return cls(mode=str(mode), local_model=str(raw.get("local_model") or ""),
                   provider_id=str(raw.get("provider_id") or ""))

    @property
    def api_provider_id(self) -> str:
        return self.provider_id or API_VISION_PROVIDER_ID


def get_seat() -> Seat:
    from finetune_studio.models.manager import ModelManager

    try:
        raw = json.loads(ModelManager.get_state(VISION_SEAT_KEY, "") or "{}")
    except ValueError:
        raw = {}
    return Seat.from_dict(raw if isinstance(raw, dict) else {})


def resolve_seat(seat: Seat | None = None) -> Seat:
    """``seat`` with ``assistant`` replaced by the main assistant's connection (local GGUF or API)."""
    from finetune_studio.models.assistant import assistant_provider_id
    from finetune_studio.models.manager import get_manager

    seat = seat or get_seat()
    if seat.mode != "assistant":
        return seat
    row = get_manager().get_provider(assistant_provider_id()) or {}
    if row.get("kind") == "openai_compat":
        return Seat(mode="api", provider_id=str(row["id"]))
    return Seat(mode="local", local_model=str(row.get("model_id") or ""))


def _check(seat: Seat) -> None:
    from finetune_studio.models.manager import get_manager

    if seat.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if seat.mode == "local":
        if not seat.local_model or not Path(seat.local_model).is_file():
            raise ValueError("choose a GGUF file that exists")
        if not mmproj_for(seat.local_model):
            raise ValueError(f"{Path(seat.local_model).name} has no mmproj*.gguf next to it: it cannot see images")
    if seat.mode == "api":
        row = get_manager().get_provider(seat.api_provider_id)
        if not row or row.get("kind") != "openai_compat":
            raise ValueError("choose an API connection for reading pictures (Settings -> AI assistant -> Connections)")
        if not row.get("model_id") or not row.get("api_key_set"):
            raise ValueError(f"the connection {row.get('name') or row['id']} has no model or no saved key")


def set_seat(seat: Seat) -> None:
    from finetune_studio.models.manager import ModelManager

    _check(seat)
    if seat.mode == "assistant":
        _check(resolve_seat(seat))      # a local assistant without an mmproj cannot read pictures: say so now
    ModelManager.set_state(VISION_SEAT_KEY, seat.as_json())


def setting_for_jobs(seat: Seat | None = None) -> str:
    """The seat as the Jobs table shows it: 'off', '' (main assistant), a connection id or 'local-file'."""
    seat = seat or get_seat()
    if seat.mode == "off":
        return "off"
    if seat.mode == "assistant":
        return ""
    if seat.mode == "api":
        return seat.api_provider_id
    from finetune_studio.models.helper import paths_match
    from finetune_studio.models.manager import get_manager

    for row in get_manager().list_providers():
        if row.get("kind") == "local_gguf" and paths_match(row.get("model_id"), seat.local_model):
            return str(row["id"])
    return "local-file"


def seat_from_setting(setting: str, local_model: str = "") -> Seat:
    """The inverse of ``setting_for_jobs`` (a local connection becomes a local seat on its GGUF)."""
    from finetune_studio.models.manager import get_manager

    setting = (setting or "").strip()
    if setting == "off":
        return Seat()
    if setting == "":
        return Seat(mode="assistant")
    if setting == "local-file":
        return Seat(mode="local", local_model=local_model.strip())
    row = get_manager().get_provider(setting)
    if row is None:
        raise ValueError(f"unknown connection {setting!r}")
    if row.get("kind") == "openai_compat":
        return Seat(mode="api", provider_id=setting)
    return Seat(mode="local", local_model=str(row.get("model_id") or ""))


def mmproj_for(gguf_path: str) -> str:
    """The vision projector the loader would pair with this GGUF (same rule as ``llama_loader``), or ``""``."""
    d = Path(gguf_path).parent
    for cand in sorted(d.glob("mmproj*.gguf")) + sorted(d.glob("*mmproj*.gguf")):
        return str(cand)
    return ""


def describe() -> dict[str, Any]:
    """Public state for Settings and job records (never a key)."""
    from finetune_studio.models.assistant import connection_label
    from finetune_studio.models.manager import get_manager

    seat = get_seat()
    real = resolve_seat(seat)
    api = (get_manager().get_provider(real.api_provider_id) or {}) if real.mode == "api" else {}
    if real.mode == "local":
        label = f"Local · {Path(real.local_model).name}"
    elif real.mode == "api":
        label = f"API · {connection_label(api) or real.api_provider_id}"
    else:
        label = "Off (OCR only)"
    if seat.mode == "assistant":
        label = f"Main assistant ({label})"
    return {"mode": seat.mode, "local_model": seat.local_model, "provider_id": seat.provider_id, "label": label,
            "setting": setting_for_jobs(seat)}


def fit_image(data: bytes, max_side: int = MAX_SIDE_PX) -> bytes:
    """The picture scaled down so its longer side is at most ``max_side`` (PNG); unchanged when already small."""
    import io

    from PIL import Image

    try:
        im = Image.open(io.BytesIO(data))
    except Exception:  # noqa: BLE001 - PIL raises several types; an undecodable picture goes as is, the model reports it
        return data
    with im:
        if max(im.size) <= max_side:
            return data
        scale = max_side / max(im.size)
        out = io.BytesIO()
        im.convert("RGB").resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS).save(out, "PNG")
        return out.getvalue()


def image_message(prompt: str, data: bytes) -> list[dict[str, Any]]:
    """One user turn with the picture as an OpenAI-style data URL (the API and llama.cpp's MTMD handler both take it)."""
    data = fit_image(data)
    mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                         {"type": "text", "text": prompt}]}]


class Reader:
    """Reads figures with the seated vision model. ``close()`` unloads a local model this reader loaded."""

    def __init__(self, seat: Seat | None = None) -> None:
        self.seat = resolve_seat(seat or get_seat())
        self._lock = threading.Lock()   # one picture at a time on a local GPU; an API takes a few at once anyway
        self._loaded_here = False
        self._api = None
        if self.seat.mode == "off":
            raise VisionError("reading pictures is off (Settings -> AI assistant -> Jobs -> Reading pictures)")
        if self.seat.mode == "local" and not mmproj_for(self.seat.local_model):
            raise VisionError(f"{Path(self.seat.local_model).name or 'the local model'} has no mmproj next to it: "
                              "it cannot see images")
        if self.seat.mode == "api":
            self._api = self._api_provider(self.seat.api_provider_id)

    @property
    def label(self) -> str:
        if self.seat.mode == "local":
            return Path(self.seat.local_model).name
        return str(self._api.config.model_id) if self._api is not None else "?"

    @staticmethod
    def _api_provider(pid: str):
        from finetune_studio.models.assistant import connection_client
        from finetune_studio.models.manager import get_manager

        row = get_manager().get_provider(pid)
        if not row or row.get("kind") != "openai_compat":
            raise VisionError(f"the connection {pid!r} for reading pictures is gone: Settings -> AI assistant")
        if not row.get("api_key_set"):
            raise VisionError(f"the connection {row.get('name') or pid} has no saved key: Settings -> AI assistant")
        return connection_client(pid).provider

    def _engine(self):
        from finetune_studio.models.helper import (
            get_configured_helper_provider,
            paths_match,
        )
        from finetune_studio.models.manager import get_manager

        engine = get_manager().engine
        want = self.seat.local_model
        loaded = str(getattr(engine, "model_path", "") or "")
        if loaded and paths_match(loaded, want):
            if not getattr(engine, "vision", False):
                raise VisionError(f"{Path(want).name} is loaded without its mmproj: reload it to read pictures")
            return engine
        helper = get_configured_helper_provider() or {}
        if helper.get("kind") == "local_gguf" and loaded and paths_match(loaded, helper.get("model_id")):
            raise VisionError(
                f"the local helper {Path(loaded).name} holds the one inference worker; reading pictures with "
                f"{Path(want).name} would unload it in the middle of mining. Pick the helper's own GGUF as the Vision "
                "reader (it needs an mmproj), use an API vision model, or seat the API helper.")
        log.info("vision reader: loading %s", want)
        engine.load(want, n_ctx=LOCAL_N_CTX)
        self._loaded_here = True
        if not getattr(engine, "vision", False):
            raise VisionError(f"{Path(want).name} loaded without vision (mmproj failed to load): see the worker log")
        return engine

    def read(self, data: bytes, title: str, where: str) -> str:
        """The model's transcription of one picture. Raises ``VisionError`` with the reason on failure."""
        from finetune_studio.models.assistant import instruct

        messages = instruct("vision", image_message(READ_PROMPT.format(title=title, where=where), data))
        try:
            if self._api is not None:
                return clean(self._api.chat(messages, max_tokens=MAX_TOKENS, temperature=0.0, top_p=1.0))
            with self._lock:
                engine = self._engine()
                out = engine.model.create_chat_completion(messages=messages, max_tokens=MAX_TOKENS, temperature=0.0)
                return clean(str(out["choices"][0]["message"]["content"] or ""))
        except VisionError:
            raise
        except Exception as e:  # provider/worker errors become one reason the job records
            raise VisionError(f"{type(e).__name__}: {e}") from e

    def close(self) -> None:
        if self._loaded_here:
            try:
                from finetune_studio.models.manager import get_manager
                get_manager().engine.unload()
            except Exception as e:  # noqa: BLE001 - unload is best effort; the worker reports its own state
                log.warning("vision reader: unload failed: %s", e)
            self._loaded_here = False
