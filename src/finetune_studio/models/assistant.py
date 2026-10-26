"""The main assistant, its connections and the jobs it does.

Connections (Settings -> AI assistant) are independent: the local model on this machine plus any number of
OpenAI-compatible API connections, each a ``model_providers`` row with its own encrypted key (``access.vault``).
Adding, testing or removing one never touches another.

The **main assistant** is one connection (``app_state`` key ``helper_seat``, kept from the helper-seat era). Every
helper job runs on it unless the job names another connection:

- ``guide``      the Guide panel in the corner of every page
- ``mining``     mining question/answer pairs from files
- ``study``      study-set rows (rephrasings, fact sheets)
- ``quiz``       reworded held-out quizzes (Testing)
- ``preference`` preference pairs (chosen / rejected answers)
- ``judge``      judging saved test answers
- ``teacher``    answers for use-case rows that have none
- ``vision``     reading pictures in documents (default **off**: pictures leave the machine only on purpose)

Per job the user may add instructions; they reach the model as a system message in front of the job's own prompt
(``instruct``). Empty by default, so a job's built-in prompt is unchanged until the user writes something.

API connections are called through their own client (``connection_client``), never through ``ModelManager``'s one
active slot: asking the Guide something with an API assistant no longer unloads the local model someone is using.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from typing import Any

from finetune_studio.models.helper import (
    DEFAULT_HELPER_PROVIDER_ID,
    HELPER_SEAT_KEY,
)

JOBS_KEY = "assistant_jobs"
OFF = "off"                     # only the vision job may be switched off
ASSISTANT = ""                  # a job setting of "" = use the main assistant
MAX_INSTRUCTIONS = 4000


@dataclass(frozen=True)
class Job:
    id: str
    label: str
    does: str                   # one line: what the job asks the model to do (shown on Settings)
    sends: str                  # what leaves the machine when the job runs on an API connection
    default: str = ASSISTANT    # ASSISTANT or OFF
    api_only: bool = False      # a local GGUF would replace the model on the card mid-job


JOBS: tuple[Job, ...] = (
    Job("guide", "Guide chat", "Answers questions in the Guide panel, checks your project with its tools and "
        "opens pages or fills fields for you.", "your questions and the tool results about your project"),
    Job("mining", "Mining Q&A pairs", "Reads each file passage and writes grounded question/answer pairs for review.",
        "the text of the documents you mine"),
    Job("study", "Study sets", "Rephrases approved pairs and writes short fact sheets so the model learns the facts, "
        "not one wording.", "your approved pairs and their source passages"),
    Job("quiz", "Reworded quizzes", "Rewrites held-out questions in new words to test real recall.",
        "the questions and answers of the dataset you quiz"),
    Job("preference", "Preference pairs", "Writes a worse answer next to each approved one for preference training.",
        "your approved pairs and their source passages"),
    Job("judge", "Test judge", "Decides whether each saved test answer is correct against the expected one.",
        "the test questions, expected answers and the model's answers"),
    Job("teacher", "Use-case teacher", "Answers use-case rows that have no answer yet, so a small model can learn "
        "from a bigger one.", "the use-case rows you upload", api_only=True),
    Job("vision", "Reading pictures", "Transcribes pictures in documents (scans, charts, table screenshots) while "
        "mining; off = OCR text only.", "the pictures in the documents you mine", default=OFF),
)
JOB_IDS: tuple[str, ...] = tuple(j.id for j in JOBS)


def job(job_id: str) -> Job:
    for j in JOBS:
        if j.id == job_id:
            return j
    raise ValueError(f"unknown assistant job {job_id!r} (one of {', '.join(JOB_IDS)})")


# ── settings (app_state) ────────────────────────────────────────────────────

def _raw_jobs() -> dict[str, Any]:
    from finetune_studio.models.manager import ModelManager

    try:
        raw = json.loads(ModelManager.get_state(JOBS_KEY, "") or "{}")
    except ValueError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _load_jobs() -> dict[str, dict[str, str]]:
    """Every job's choice and instructions; a job the user never saved gets its default (assistant, vision off)."""
    raw = _raw_jobs()
    out: dict[str, dict[str, str]] = {}
    for spec in JOBS:
        entry = raw.get(spec.id) if isinstance(raw.get(spec.id), dict) else {}
        out[spec.id] = {"provider_id": str(entry.get("provider_id", spec.default) or ""),
                        "instructions": str(entry.get("instructions") or "")}
    return out


def assistant_provider_id() -> str:
    """The connection that is the main assistant (a deleted row falls back to the local model)."""
    from finetune_studio.models.manager import ModelManager, get_manager

    pid = ModelManager.get_state(HELPER_SEAT_KEY, DEFAULT_HELPER_PROVIDER_ID) or DEFAULT_HELPER_PROVIDER_ID
    if pid != DEFAULT_HELPER_PROVIDER_ID and get_manager().get_provider(pid) is None:
        return DEFAULT_HELPER_PROVIDER_ID
    return pid


def set_assistant_provider_id(pid: str) -> None:
    from finetune_studio.models.manager import ModelManager, get_manager

    if get_manager().get_provider(pid) is None:
        raise ValueError(f"unknown connection {pid!r}")
    ModelManager.set_state(HELPER_SEAT_KEY, pid)


def job_setting(job_id: str) -> str:
    """The stored choice for a job: ``""`` (main assistant), ``"off"`` or a connection id.

    Vision keeps its own seat (``models.vision``; it may also be ``"local-file"``, a vision GGUF that is not a
    connection)."""
    job(job_id)
    if job_id == "vision":
        from finetune_studio.models.vision import setting_for_jobs

        return setting_for_jobs()
    return _load_jobs()[job_id]["provider_id"]


def job_provider_id(job_id: str) -> str:
    """The connection a job runs on right now: its own choice when that row exists, else the main assistant.

    ``"off"`` for a job that is switched off (vision only)."""
    from finetune_studio.models.manager import get_manager

    choice = job_setting(job_id)
    if choice == OFF:
        return OFF
    if choice == "local-file":
        return "local-file"
    if choice and get_manager().get_provider(choice) is not None:
        return choice
    return assistant_provider_id()


def job_instructions(job_id: str) -> str:
    job(job_id)
    return _load_jobs()[job_id]["instructions"].strip()


def instructions_tag(job_id: str) -> str:
    """Short hash of the job's extra instructions ('' when none), stored with mining jobs as provenance."""
    text = job_instructions(job_id)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12] if text else ""


def set_job(job_id: str, *, provider_id: str | None = None, instructions: str | None = None,
            local_model: str = "") -> None:
    """Save a job's connection (``""`` = main assistant, ``"off"`` for vision, or a connection id) and/or its
    extra instructions. Vision also takes ``"local-file"`` with ``local_model`` (a GGUF with an mmproj)."""
    from finetune_studio.models.manager import ModelManager, get_manager

    spec = job(job_id)
    raw = _raw_jobs()
    entry = dict(raw.get(job_id) or {}) if isinstance(raw.get(job_id), dict) else {}
    if provider_id is not None and job_id == "vision":
        from finetune_studio.models import vision

        vision.set_seat(vision.seat_from_setting(provider_id, local_model))
    elif provider_id is not None:
        pid = provider_id.strip()
        if pid == OFF and spec.default != OFF:
            raise ValueError(f"{spec.label} cannot be switched off")
        if pid not in (ASSISTANT, OFF):
            row = get_manager().get_provider(pid)
            if row is None:
                raise ValueError(f"unknown connection {pid!r}")
            if spec.api_only and row.get("kind") != "openai_compat":
                raise ValueError(f"{spec.label} needs an API connection (a local model would replace the one on the card)")
        entry["provider_id"] = pid
    if instructions is not None:
        text = instructions.strip()
        if len(text) > MAX_INSTRUCTIONS:
            raise ValueError(f"instructions are limited to {MAX_INSTRUCTIONS} characters")
        entry["instructions"] = text
    raw[job_id] = entry
    ModelManager.set_state(JOBS_KEY, json.dumps(raw, ensure_ascii=False))


def forget_connection(pid: str) -> None:
    """A removed connection: jobs that named it go back to the main assistant (vision goes back to off)."""
    from finetune_studio.models import vision
    from finetune_studio.models.manager import ModelManager

    seat = vision.get_seat()
    if seat.mode == "api" and seat.api_provider_id == pid:
        ModelManager.set_state(vision.VISION_SEAT_KEY, vision.Seat().as_json())   # pictures go back to OCR only
    raw = _raw_jobs()
    changed = False
    for jid, entry in list(raw.items()):
        if isinstance(entry, dict) and entry.get("provider_id") == pid:
            if job(jid).default == OFF:
                entry.pop("provider_id", None)
            else:
                entry["provider_id"] = ASSISTANT
            changed = True
    if changed:
        ModelManager.set_state(JOBS_KEY, json.dumps(raw, ensure_ascii=False))


def instruct(job_id: str, messages: list[dict[str, Any]], text: str | None = None) -> list[dict[str, Any]]:
    """``messages`` with the job's extra instructions in front as a system message (unchanged when there are none).

    ``text`` overrides the saved instructions (a mining job applies the instructions recorded when it was queued)."""
    text = job_instructions(job_id) if text is None else text.strip()
    if not text:
        return messages
    note = f"Additional instructions from the user for this job ({job(job_id).label}):\n{text}"
    if messages and messages[0].get("role") == "system":
        return [{**messages[0], "content": f"{messages[0].get('content') or ''}\n\n{note}"}, *messages[1:]]
    return [{"role": "system", "content": note}, *messages]


def jobs_state() -> list[dict[str, Any]]:
    """Every job with its saved choice, the connection it resolves to and its instructions (Settings)."""
    from finetune_studio.models.manager import get_manager

    mgr = get_manager()
    stored = _load_jobs()
    out = []
    for spec in JOBS:
        pid = job_provider_id(spec.id)
        row = mgr.get_provider(pid) if pid not in (OFF, "local-file") else None
        label = connection_label(row) if row else ("Off" if pid == OFF else "")
        extra: dict[str, Any] = {}
        if spec.id == "vision":
            from finetune_studio.models import vision

            seat = vision.get_seat()
            label = vision.describe()["label"]
            extra = {"local_model": seat.local_model}
        out.append({**asdict(spec), "setting": job_setting(spec.id), "instructions": stored[spec.id]["instructions"],
                    "provider_id": pid, "provider_label": label, "provider_kind": (row or {}).get("kind", ""), **extra})
    return out


def connection_label(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    if row.get("kind") == "openai_compat":
        name = str(row.get("name") or row.get("id") or "API")
        model = str(row.get("model_id") or "model not set")
        return name if model in name else f"{name} · {model}"
    return str(row.get("label") or row.get("name") or row.get("id") or "")


# ── API connection clients (independent of ModelManager's one active slot) ──

_clients: dict[str, tuple[str, Any]] = {}
_clients_lock = threading.Lock()


class ConnectionClient:
    """One API connection as a chat backend. Duck-types the parts of ``ModelManager`` the helper code calls
    (``chat``, ``generate``, ``active``), so a job can use it where it used the manager, without loading anything."""

    concurrent = True

    def __init__(self, row: dict[str, Any], provider: Any) -> None:
        self.row = row
        self.provider = provider

    def chat(self, messages: list[dict[str, Any]], **gen: Any) -> str:
        return self.provider.chat(messages, **gen)

    def generate(self, prompt: str, **gen: Any) -> str:
        return self.provider.generate(prompt, **gen)

    def active(self) -> dict[str, Any]:
        return {"id": self.row["id"], "kind": "openai_compat", "name": self.row.get("name") or self.row["id"],
                "model_id": self.row.get("model_id") or "", "label": connection_label(self.row), "loaded": True}


def connection_client(pid: str) -> ConnectionClient:
    """A ready client for an API connection, rebuilt when its URL, model, settings or key changed."""
    from finetune_studio.models.manager import ModelManager, get_manager
    from finetune_studio.models.providers import ProviderConfig, build_provider

    row = get_manager().get_provider(pid)
    if row is None or row.get("kind") != "openai_compat":
        raise ValueError(f"{pid!r} is not an API connection (Settings -> AI assistant)")
    key = ModelManager._api_key_for(pid)
    stamp = hashlib.sha256(json.dumps([row.get("base_url"), row.get("model_id"), row.get("extra"), key],
                                      sort_keys=True, default=str).encode()).hexdigest()
    with _clients_lock:
        cached = _clients.get(pid)
        if cached and cached[0] == stamp:
            return ConnectionClient(row, cached[1])
        if cached:
            cached[1].unload()
        provider = build_provider(ProviderConfig(
            id=row["id"], name=row.get("name") or row["id"], kind="openai_compat", model_id=row.get("model_id") or "",
            base_url=row.get("base_url") or "", api_key=key, extra=dict(row.get("extra") or {})))
        provider.load()
        _clients[pid] = (stamp, provider)
        return ConnectionClient(row, provider)


def drop_client(pid: str) -> None:
    with _clients_lock:
        cached = _clients.pop(pid, None)
    if cached:
        cached[1].unload()
