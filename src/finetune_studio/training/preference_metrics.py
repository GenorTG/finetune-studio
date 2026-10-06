"""Live preference-training metrics (DPO ``rewards/*``, KTO ``rewards/*`` + ``kl``).

TRL logs these next to ``loss``. They are the only evidence a preference run is
learning anything: ``loss`` alone sits at ``ln 2 ≈ 0.693`` (DPO) whether the policy
moved or not, while ``rewards/accuracies`` (share of pairs where the policy ranks
the chosen answer above the rejected one, relative to the reference) and
``rewards/margins`` move as soon as the adapter does.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# TRL key -> short label used in step-log lines.
_TRAIN_KEYS: dict[str, str] = {
    "rewards/accuracies": "acc",
    "rewards/margins": "margin",
    "rewards/chosen": "r_chosen",
    "rewards/rejected": "r_rejected",
    "kl": "kl",
}


def extract_preference_metrics(logs: Mapping[str, Any]) -> dict[str, float]:
    """Preference metrics present in one ``on_log`` payload (train keys + ``eval_`` twins).

    Empty for SFT logs, so callers can merge the result unconditionally.
    """
    out: dict[str, float] = {}
    for key in _TRAIN_KEYS:
        for prefix in ("", "eval_"):
            value = logs.get(prefix + key)
            if isinstance(value, (int, float)):
                out[prefix + key] = round(float(value), 4)
    return out


def format_metric_suffix(metrics: Mapping[str, float], *, prefix: str = "") -> str:
    """`` | acc=0.75 | margin=0.12`` for the step log; empty when nothing was logged."""
    parts = [
        f"{label}={metrics[prefix + key]}"
        for key, label in _TRAIN_KEYS.items()
        if (prefix + key) in metrics and key in ("rewards/accuracies", "rewards/margins", "kl")
    ]
    return "".join(f" | {part}" for part in parts)
