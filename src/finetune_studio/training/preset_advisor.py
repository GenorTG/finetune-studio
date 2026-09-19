"""Training preset advisor — propose settings from base-model size + dataset size.

WHAT THIS DOES
==============
The user picks a quality tier (smoke / balanced / precision / overkill), a base
model and a dataset. This module turns that into concrete hyperparameters with
the math shown, instead of one fixed template for every situation.

Principles (evidence-backed, see repo history):
- Domain recall is governed by real optimizer steps, not epochs.
  A 515-pair factual dataset reached 95.1% strict recall at 772 optimizer
  steps (12 ep x 515 / (2*4)); the same rank at 257 steps only hit 67%.
- Rank should scale with what the model must absorb, tempered by base size.
- Learning rate should fall as base size grows (big models destabilize fast).
- Nothing here is a cage: every value is a proposal with the arithmetic
  attached, and the caller can override any field.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Tier anchors for the reference case: ~500 pairs, ~4B-param base.
# They reproduce the observed evidence points: 4ep/r128 -> 67%, 12ep/r128 -> 95%.
_TIER_ANCHORS: dict[str, dict[str, Any]] = {
    "smoke": {"epochs": 1, "rank": 16, "alpha_mult": 2.0, "lr": "2e-4",
              "steps_floor": 0, "blurb": "plumbing check, not memorization"},
    "balanced": {"epochs": 6, "rank": 64, "alpha_mult": 2.0, "lr": "2e-4",
                 "steps_floor": 400, "blurb": "solid recall on typical domain datasets"},
    "precision": {"epochs": 12, "rank": 128, "alpha_mult": 2.0, "lr": "2e-4",
                  "steps_floor": 700, "blurb": "the 95.1%-strict evidence class"},
    "overkill": {"epochs": 24, "rank": 256, "alpha_mult": 2.0, "lr": "1e-4",
                 "steps_floor": 1000, "blurb": "only after Precision fails on a verified dataset"},
}

# Rank scaling by base parameter count (billions). Reference point: ~4B -> 1.0x.
def _rank_factor(params_b: float) -> float:
    if params_b <= 1.0:
        return 0.5
    if params_b <= 2.0:
        return 0.75
    if params_b <= 8.0:
        return 1.0
    return 1.0  # bigger bases: do not inflate rank; steps + LR do the work


def _lr_for(params_b: float, tier_lr: str) -> str:
    if params_b <= 7.0:
        return tier_lr
    if params_b <= 15.0:
        return "1e-4"
    return "8e-5"


_PARAM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*[bB](?![A-Za-z0-9])")


def guess_base_params_b(model_ref: str) -> tuple[float | None, str]:
    """Best-effort parameter count (in billions) from a model name/path.

    Returns (params_b, source). Handles "Qwen3-4B", "Qwen__Qwen3-0.6B",
    "27B-abliterated-Q4_K_M.gguf", "8.3B" etc. GGUF quant suffixes
    (Q4_K_M) are not confused with size because the [bB] unit is required.
    """
    if not model_ref:
        return None, "unknown"
    name = model_ref.replace("__", "/").rstrip("/")
    leaf = Path(name).name
    m = _PARAM_RE.search(leaf)
    if m:
        try:
            return float(m.group(1).replace(",", ".")), "name"
        except ValueError:
            pass
    m = _PARAM_RE.search(name)
    if m:
        try:
            return float(m.group(1).replace(",", ".")), "name"
        except ValueError:
            pass
    return None, "unknown"


@dataclass
class Advisory:
    tier: str
    base_params_b: float | None
    size_source: str
    pair_count: int
    avg_chars_per_pair: float
    effective_batch: int
    num_epochs: int
    lora_rank: int
    lora_alpha: int
    learning_rate: str
    batch_size: int
    gradient_accumulation_steps: int
    optimizer_steps: int
    steps_floor: int
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier, "base_params_b": self.base_params_b,
            "size_source": self.size_source, "pair_count": self.pair_count,
            "avg_chars_per_pair": round(self.avg_chars_per_pair, 1),
            "effective_batch": self.effective_batch,
            "num_epochs": self.num_epochs, "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha, "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer_steps": self.optimizer_steps,
            "steps_floor": self.steps_floor,
            "notes": self.notes, "warnings": self.warnings,
        }


def dataset_stats(jsonl_path: str | Path) -> tuple[int, float]:
    """(pair_count, avg_chars_per_pair) for a ShareGPT/openai JSONL file."""
    p = Path(jsonl_path)
    if not p.exists():
        return 0, 0.0
    n = 0
    total_chars = 0
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = row.get("messages") or row.get("conversations") or []
            if isinstance(msgs, list):
                total_chars += sum(len(str(m.get("content", "") or "")) for m in msgs)
            n += 1
    return n, (total_chars / n) if n else 0.0


def propose(
    *,
    tier: str,
    base_model_ref: str,
    dataset_path: str | None = None,
    pair_count_hint: int | None = None,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
) -> Advisory:
    """Compute recommended settings for a base model + dataset at a tier.

    Pair count comes from the dataset file when available, else from
    ``pair_count_hint``; with neither, the reference 500-pair assumption is
    used and flagged as an assumption.
    """
    anchor = _TIER_ANCHORS.get(tier)
    if anchor is None:
        raise ValueError(f"Unknown tier: {tier}")

    params_b, size_source = guess_base_params_b(base_model_ref)
    notes: list[str] = []
    warnings: list[str] = []

    if params_b is None:
        params_b = 4.0
        size_source = "assumed"
        warnings.append(
            "Could not read the model size from its name — assumed a ~4B-class "
            "base. Rank/LR scale with size, so double-check the pick."
        )
        notes.append("Assumed base size: 4B.")
    else:
        notes.append(f"Base size parsed from name: {params_b:g}B parameters.")

    if pair_count_hint is not None:
        pair_count = max(0, int(pair_count_hint))
        avg_chars = 0.0
    elif dataset_path:
        pair_count, avg_chars = dataset_stats(dataset_path)
        if pair_count == 0:
            warnings.append("Dataset file empty or unreadable — using the 500-pair reference assumption.")
            pair_count = 500
    else:
        pair_count = 500
        avg_chars = 0.0
        warnings.append("No dataset selected yet — sized for the ~500-pair reference case; re-check after picking one.")
    notes.append(f"Dataset: {pair_count} pairs.")

    eff_batch = max(1, batch_size * gradient_accumulation_steps)

    # Dataset-size scaling: reference is ~500 pairs. Sub-linear exponent keeps
    # big datasets from collapsing to 1 epoch while small ones scale up enough
    # to accumulate real optimizer steps.
    ds_factor = math.sqrt(500.0 / max(pair_count, 1))
    ds_factor = min(max(ds_factor, 0.25), 4.0)
    size_factor = (_rank_factor(params_b))
    epochs = anchor["epochs"] * ds_factor

    # Steps floor: recall quality is governed by real optimizer steps.
    # 772 steps @ r128 = 95.1% strict on the reference dataset; 257 steps = 67%.
    floor = anchor["steps_floor"]
    min_epochs_for_floor = math.ceil(floor * eff_batch / max(pair_count, 1)) if floor else 0
    if min_epochs_for_floor > epochs:
        epochs = min_epochs_for_floor
        notes.append(
            f"Epochs raised to {epochs} so the run clears the {floor}-optimizer-step "
            f"floor for this tier ({pair_count} pairs x epochs / {eff_batch} effective batch)."
        )

    epochs = int(min(max(round(epochs), 1), 60))

    rank = int(anchor["rank"] * size_factor)
    rank = max(8, min(rank, 256))
    alpha = int(rank * anchor["alpha_mult"])

    lr = _lr_for(params_b, anchor["lr"])
    if size_source == "assumed":
        pass
    elif params_b > 7.0:
        notes.append(f"LR lowered to {lr} for a {params_b:g}B base.")
    if params_b <= 1.0:
        notes.append(f"Rank scaled down to {rank} — tiny bases saturate small ranks.")

    steps = (pair_count * epochs) // eff_batch
    if tier != "smoke" and steps < floor:
        warnings.append(
            f"Configuration yields only ~{steps} optimizer steps (floor for this tier: {floor}). "
            "Raise epochs or reduce effective batch."
        )
    notes.append(
        f"Step math: {pair_count} pairs x {epochs} epochs / {eff_batch} "
        f"effective batch = ~{steps} optimizer steps."
    )
    notes.append(anchor["blurb"] + ".")

    return Advisory(
        tier=tier, base_params_b=params_b, size_source=size_source,
        pair_count=pair_count, avg_chars_per_pair=avg_chars,
        effective_batch=eff_batch, num_epochs=epochs, lora_rank=rank,
        lora_alpha=alpha, learning_rate=lr, batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimizer_steps=steps, steps_floor=floor,
        notes=notes, warnings=warnings,
    )
