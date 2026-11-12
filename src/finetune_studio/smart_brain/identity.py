"""A project's Smart Brain identity: who it speaks for, how it sounds and how it answers.

Stored per project (``projects/<pid>/brain_identity.json``), outside the index directory: a rebuild from scratch swaps
that directory. ``compose`` turns it into the instructions the answer model gets in the studio's chat, in Testing and
in the exported package, so what is tested is what ships. An empty identity composes to "" and the default grounded
prompt applies. The grounding rule comes last and is not editable: a personality can change the voice, not the facts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from finetune_studio.data.fs.paths import project_dir

LIMITS = {"name": 200, "about": 20_000, "personality": 4_000, "answer_rules": 8_000}
FIELDS = tuple(LIMITS)
GROUNDING = (
    "Answer questions of fact using ONLY the ABOUT section and the CONTEXT passages. If they do not say, say so; never "
    "guess. Your voice may follow the instructions above, but never change what the sources say. Quote the source "
    "filename in [brackets] when relevant."
)


class IdentityError(ValueError):
    """A field is the wrong type or too long; the message says which."""


def _path(pid: str) -> Path:
    return project_dir(pid) / "brain_identity.json"


def clean(data: dict[str, Any]) -> dict[str, str]:
    """Only the known fields, trimmed. Raises ``IdentityError`` naming a field that is not text or is too long."""
    out: dict[str, str] = {}
    for key, limit in LIMITS.items():
        value = data.get(key, "")
        if not isinstance(value, str):
            raise IdentityError(f"{key} must be text")
        value = value.strip()
        if len(value) > limit:
            raise IdentityError(f"{key} is {len(value):,} characters; the limit is {limit:,}")
        out[key] = value
    return out


def load(pid: str) -> dict[str, str]:
    """The saved identity; every field empty when none was saved or the file cannot be read."""
    try:
        return clean(json.loads(_path(pid).read_text(encoding="utf-8")))
    except (OSError, ValueError, AttributeError):
        return dict.fromkeys(FIELDS, "")


def save(pid: str, data: dict[str, Any]) -> dict[str, str]:
    """Replace the identity (atomic); returns what was stored."""
    identity = clean(data)
    path = _path(pid)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return identity


def is_set(identity: dict[str, str]) -> bool:
    return any(identity.get(key) for key in FIELDS)


def compose(identity: dict[str, str]) -> str:
    """The instructions for the answer model; "" when the identity is empty."""
    if not is_set(identity):
        return ""
    parts: list[str] = []
    if identity.get("name"):
        parts.append(f"You are {identity['name']}.")
    if identity.get("about"):
        parts.append(f"ABOUT:\n{identity['about']}")
    if identity.get("personality"):
        parts.append(f"HOW YOU SOUND:\n{identity['personality']}")
    if identity.get("answer_rules"):
        parts.append(f"HOW TO ANSWER:\n{identity['answer_rules']}")
    parts.append(GROUNDING)
    return "\n\n".join(parts)


def system_prompt(pid: str) -> str:
    """``compose`` of the project's saved identity."""
    return compose(load(pid))


def adopt_imported(pid: str, corpus_dir: Path) -> bool:
    """A bundle's ``identity.json``: saved as the project's identity unless it already has one; the file is removed."""
    bundled = Path(corpus_dir) / "identity.json"
    if not bundled.is_file():
        return False
    try:
        data = json.loads(bundled.read_text(encoding="utf-8"))
        if is_set(load(pid)) or not isinstance(data, dict):
            return False
        save(pid, data)
        return True
    except (OSError, ValueError):
        return False
    finally:
        bundled.unlink(missing_ok=True)


def package_record(pid: str) -> dict[str, Any]:
    """What an exported package carries in its (encrypted) manifest; {} when no identity is set."""
    identity = load(pid)
    return {**identity, "system_prompt": compose(identity)} if is_set(identity) else {}
