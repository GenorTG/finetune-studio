"""Which layers a LoRA adapter trains, read from the loaded model.

A hard-coded list (``q_proj … down_proj``) only fits Llama-style decoders: it misses
the linear-attention mixers of hybrid models (Qwen3.5 ``linear_attn.in_proj_qkv`` …),
GPT-2 style ``c_attn``/``c_fc`` and every other naming scheme. Evidence (LoRA Without
Regret 2025, Biderman et al. 2024) says adapters belong on every linear layer of the
language model, so this module finds them on whatever architecture was loaded:

- every ``nn.Linear`` (bitsandbytes 4/8-bit layers subclass it) and HF ``Conv1D``;
- not the output head (``get_output_embeddings``), not vision/audio towers or their
  projectors (the adapter is for text), not MoE routers (training them unbalances
  expert load), and not layers narrower than the rank (a rank-r update there is
  just a full-rank update with extra parameters).

Mixture-of-experts blocks in transformers 5 keep their experts as fused 3-D
parameters (``mlp.experts.gate_up_proj`` …), not ``nn.Linear``; those go to PEFT's
``target_parameters`` so the experts train too, not only attention.

``modules`` is the short leaf-name list (``["down_proj", "q_proj", …]``) when no
excluded linear layer shares a leaf name, else full module paths, which PEFT matches
exactly. Either form is valid ``LoraConfig.target_modules``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A module path segment containing one of these belongs to a non-text tower.
_NON_TEXT_PARTS = ("vision", "visual", "image", "audio", "speech", "projector", "patch_embed")
# MoE router leaf names (dense MLPs use ``gate_proj``, never a bare ``gate``).
_ROUTER_LEAVES = frozenset({"gate", "router", "shared_expert_gate"})


def _linear_dims(module: Any) -> tuple[int, int] | None:
    """(in, out) for a Linear-like module, None for anything else."""
    import torch

    if isinstance(module, torch.nn.Linear):
        return module.in_features, module.out_features
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:  # pragma: no cover - transformers always ships it
        return None
    if isinstance(module, Conv1D):
        return module.weight.shape[0], module.nf
    return None


def _skip_reason(name: str, dims: tuple[int, int], rank: int) -> str | None:
    leaf = name.rsplit(".", 1)[-1]
    segments = name.lower().split(".")
    if any(part in seg for seg in segments for part in _NON_TEXT_PARTS):
        return "non-text tower"
    if leaf in _ROUTER_LEAVES or "router" in leaf:
        return "MoE router"
    if min(dims) < rank:
        return f"narrower than rank {rank}"
    return None


@dataclass
class LoraTargets:
    """What ``LoraConfig`` gets: ``target_modules`` and (MoE only) ``target_parameters``."""

    modules: list[str]
    parameters: list[str] = field(default_factory=list)

    def peft_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"target_modules": self.modules}
        if self.parameters:
            kw["target_parameters"] = self.parameters
        return kw

    def describe(self) -> str:
        """One log line naming the distinct layer kinds."""
        leaves = sorted({t.rsplit(".", 1)[-1] for t in self.modules})
        line = f"LoRA targets: {', '.join(leaves)}"
        if len(leaves) != len(self.modules):
            line += f" ({len(self.modules)} layers)"
        if self.parameters:
            line += f"; MoE experts: {', '.join(self.parameters)}"
        return line


def _expert_parameters(model: Any) -> list[str]:
    """``<module leaf>.<param>`` for fused 3-D expert weights outside non-text towers."""
    found: set[str] = set()
    for name, module in model.named_modules():
        if not name or not type(module).__name__.endswith("Experts"):
            continue
        if any(part in seg for seg in name.lower().split(".") for part in _NON_TEXT_PARTS):
            continue
        leaf = name.rsplit(".", 1)[-1]
        for pname, param in module.named_parameters(recurse=False):
            if param.dim() == 3:
                found.add(f"{leaf}.{pname}")
    return sorted(found)


def discover_lora_targets(model: Any, rank: int) -> LoraTargets:
    """Every language-model linear layer of ``model`` plus fused MoE experts (see module doc).

    Raises ValueError when the model has no trainable linear layer at all, so a
    training run fails with a reason instead of PEFT's generic "no modules found".
    """
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    picked: list[str] = []
    skipped: list[str] = []
    for name, module in model.named_modules():
        if not name or module is head:
            continue
        dims = _linear_dims(module)
        if dims is None:
            continue
        (skipped if _skip_reason(name, dims, rank) else picked).append(name)
    if not picked:
        raise ValueError(
            f"{type(model).__name__} has no linear layer a LoRA adapter can train "
            "(only output head, vision/audio, router or tiny layers were found)."
        )
    leaves = {n.rsplit(".", 1)[-1] for n in picked}
    modules = sorted(leaves) if leaves.isdisjoint(n.rsplit(".", 1)[-1] for n in skipped) else sorted(picked)
    return LoraTargets(modules=modules, parameters=_expert_parameters(model))


def resolve_lora_targets(model: Any, rank: int, configured: list[str] | None) -> LoraTargets:
    """The configured module list when one was given (expert override), else :func:`discover_lora_targets`."""
    if configured:
        return LoraTargets(modules=list(configured))
    return discover_lora_targets(model, rank)
