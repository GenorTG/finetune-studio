"""Distillation: the teacher (a big model on an API provider row) writes the answers the student will learn.

Only API providers can teach: a local GGUF teacher would have to replace the model on the card, and a teacher that
runs locally is rarely the "big model" this template is for (bring its answers in the file instead). The provider is
built standalone (``models.providers.build_provider``, as the mining seat does), so the manager's loaded model and the
helper seat are left alone. A failed or empty answer drops that prompt and is counted, never filled in.
"""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

TEACHER_MAX_TOKENS = 2048
TEACHER_TEMPERATURE = 0.2
PARALLEL = 4
_MAX_ERRORS_SHOWN = 5

ChatFn = Callable[[list[dict[str, str]]], str]


class TeacherUnavailable(ValueError):
    """The chosen provider cannot act as the teacher (message is shown to the user)."""


@dataclass
class TeacherReport:
    asked: int = 0
    answered: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    model: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"asked": self.asked, "answered": self.answered, "failed": self.failed, "errors": self.errors,
                "model": self.model}


def open_teacher(provider_id: str) -> tuple[ChatFn, str]:
    """(chat callable, model id) for an API provider row; :class:`TeacherUnavailable` otherwise."""
    from finetune_studio.models.assistant import connection_client, instruct
    from finetune_studio.models.manager import get_manager

    row = get_manager().get_provider(provider_id)
    if row is None:
        raise TeacherUnavailable(f"unknown provider {provider_id!r}")
    if row.get("kind") != "openai_compat":
        raise TeacherUnavailable("the teacher must be an API provider (a local GGUF would replace the model on the "
                                 "card); or put the teacher's answers in the file as teacher_answer")
    client = connection_client(provider_id)

    def chat(messages: list[dict[str, str]]) -> str:
        return client.chat(instruct("teacher", messages), max_tokens=TEACHER_MAX_TOKENS,
                           temperature=TEACHER_TEMPERATURE)

    return chat, str(row.get("model_id") or row.get("name") or provider_id)


def fill_answers(rows: list[Any], chat: ChatFn, instruction: str = "", *,
                 progress: Callable[[int, int], None] | None = None) -> tuple[list[Any], TeacherReport]:
    """Ask the teacher for every row without an answer; rows it could not answer are dropped (and counted)."""
    from finetune_studio.usecases.metrics import strip_reasoning

    report = TeacherReport()
    todo = [i for i, r in enumerate(rows) if isinstance(r, dict) and not _has_answer(r) and _prompt(r)]
    report.asked = len(todo)

    def ask(i: int) -> tuple[int, str, str]:
        msgs = ([{"role": "system", "content": instruction}] if instruction else []) + \
               [{"role": "user", "content": _prompt(rows[i])}]
        try:
            return i, strip_reasoning(chat(msgs)), ""
        except Exception as exc:  # noqa: BLE001 - provider errors vary; counted, never an answer
            return i, "", str(exc)

    answers: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=PARALLEL, thread_name_prefix="fts-teacher") as pool:
        for done, (i, text, err) in enumerate(pool.map(ask, todo), 1):
            if text:
                answers[i] = text
            else:
                report.failed += 1
                if len(report.errors) < _MAX_ERRORS_SHOWN:
                    report.errors.append(f"row {i + 1}: {err or 'empty answer'}")
            if progress is not None:
                progress(done, len(todo))
    report.answered = len(answers)
    asked = set(todo)
    out = []
    for i, r in enumerate(rows):
        if i in answers:
            out.append({**r, "teacher_answer": answers[i]})
        elif i not in asked:
            out.append(r)
    return out, report


def _has_answer(row: dict[str, Any]) -> bool:
    return any(str(row.get(k) or "").strip() for k in ("teacher_answer", "answer", "output"))


def _prompt(row: dict[str, Any]) -> str:
    for k in ("prompt", "question", "input"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""
