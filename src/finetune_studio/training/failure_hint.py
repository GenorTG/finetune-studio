"""Plain-language diagnosis of a failed training run.

The Training page shows the raw error of a failed run; for device out-of-memory
errors (``accel.is_oom_message``) this module adds a hint with concrete knobs,
computed from the run's own recorded settings. Only knobs that exist in the
Training form are offered: there is no standalone gradient-checkpointing switch
(the "Use Unsloth" option brings 4-bit weights + checkpointing where Unsloth is
installed), and the loader already retries a load-time OOM in 4-bit by itself.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from finetune_studio.accel import is_oom_message

_TOTAL_RE = re.compile(r"total capacity of ([\d.]+) (MiB|GiB)", re.IGNORECASE)
_SELF_RE = re.compile(r"this process has ([\d.]+) (MiB|GiB) memory in use", re.IGNORECASE)
_OTHER_RE = re.compile(r"Process (\d+) has ([\d.]+) (MiB|GiB) memory in use", re.IGNORECASE)
# Foreign GPU users below this are noise (a desktop compositor, a CUDA context).
_FOREIGN_MIB_MIN = 1024
_MIN_SEQ = 256


@dataclass
class Knob:
    field: str          # name of the Training-form field to change
    label: str
    current: Any
    suggested: Any
    note: str = ""


@dataclass
class FailureHint:
    kind: str
    title: str
    summary: str
    knobs: list[Knob] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mib(value: str, unit: str) -> float:
    return float(value) * (1024.0 if unit.lower() == "gib" else 1.0)


def _gib(mib: float) -> str:
    return f"{mib / 1024.0:.1f} GiB"


def _int(settings: dict[str, Any], key: str) -> int | None:
    try:
        v = int(settings.get(key))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _knobs(settings: dict[str, Any]) -> list[Knob]:
    out: list[Knob] = []
    batch, accum = _int(settings, "batch_size"), _int(settings, "gradient_accumulation_steps") or 1
    seq, rank = _int(settings, "max_seq_length"), _int(settings, "lora_rank")
    if batch and batch > 1:
        new = max(1, batch // 2)
        keep = math.ceil(batch * accum / new)
        out.append(Knob("batch_size", "Batch size", batch, new,
                        f"Halves the activations held per step. Raise gradient accumulation {accum} → {keep} "
                        "to keep the same effective batch."))
    elif batch == 1:
        out.append(Knob("batch_size", "Batch size", 1, 1, "Already 1: this knob cannot go lower."))
    if seq and seq > _MIN_SEQ:
        new = max(_MIN_SEQ, (seq // 2) // 128 * 128 or _MIN_SEQ)
        out.append(Knob("max_seq_length", "Max sequence", seq, new,
                        "Activation memory grows with sequence length; longer examples are truncated "
                        "to this many tokens, so check your longest rows first."))
    if settings.get("unsloth") in (None, False, 0, "0", ""):
        out.append(Knob("unsloth", "Use Unsloth (4-bit, faster, less VRAM)", "off", "on",
                        "4-bit base weights plus gradient checkpointing, roughly half the VRAM, "
                        "where Unsloth is installed (otherwise training runs the standard path)."))
    if rank and rank > 32:
        out.append(Knob("lora_rank", "LoRA rank", rank, 16,
                        "Smaller adapter: a modest saving, last resort."))
    return out


def _gpu_numbers(error: str) -> tuple[float | None, float | None, list[tuple[str, float]]]:
    total = _TOTAL_RE.search(error)
    own = _SELF_RE.search(error)
    others = [(pid, _mib(v, u)) for pid, v, u in _OTHER_RE.findall(error)]
    return (_mib(*total.groups()) if total else None,
            _mib(*own.groups()) if own else None, others)


def diagnose_failure(error: str, settings: dict[str, Any] | None = None) -> FailureHint | None:
    """Hint for ``error`` (a run's failure text), or None when nothing useful can be said."""
    if not error or not is_oom_message(error):
        return None
    total, own, others = _gpu_numbers(error)
    foreign = [(pid, mib) for pid, mib in others if mib >= _FOREIGN_MIB_MIN]
    if total and own:
        summary = (f"The GPU ran out of memory: it has {_gib(total)} and this run alone was holding "
                   f"{_gib(own)} when the next allocation failed.")
    else:
        summary = "The GPU ran out of memory while this run was loading or training."
    summary += " Nothing is wrong with your data; the run asked for more VRAM than the card has."
    tips: list[str] = []
    if foreign:
        held = ", ".join(f"PID {pid} ({_gib(mib)})" for pid, mib in foreign)
        tips.append(f"Other programs are holding VRAM on this GPU: {held}. Close them and retry "
                    "(the Studio never stops foreign processes for you).")
    else:
        tips.append("Close other programs that use this GPU (games, browsers with GPU rendering, "
                    "other model servers) so the whole card is free, then retry.")
    tips.append("If the base model alone is close to the card's VRAM, pick a smaller base model: "
                "no setting below can shrink the weights themselves.")
    return FailureHint(
        kind="oom", title="Out of GPU memory", summary=summary,
        knobs=_knobs(settings or {}), tips=tips,
    )
