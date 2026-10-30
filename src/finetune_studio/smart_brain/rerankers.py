"""Rerank-model adapter (cross-encoder from sentence-transformers).

Special name prefixes:
  - "reranker_local:<path>"  -> load from local CrossEncoder-saved dir
  - any other name           -> fetch from HuggingFace
"""
from __future__ import annotations

from finetune_studio.smart_brain.constants import (
    CPU_RERANKER,
    DEFAULT_RERANKER,
    RERANKER_LOCAL_PREFIX,
)


def default_reranker(device: str = "auto") -> str:
    """The reranker a new index gets: the accurate multilingual one with a GPU, the small English one without."""
    from finetune_studio.smart_brain.devices import resolve_device

    return CPU_RERANKER if resolve_device(device).split(":")[0] == "cpu" else DEFAULT_RERANKER


def half_precision_kwargs(device: str) -> dict:
    """``CrossEncoder`` kwargs for fp16 weights on a CUDA/ROCm GPU (~4x faster, same ranking), none elsewhere.

    Rerank scores only order one query's passages, so unlike the embedder (fp32: index and query vectors must
    match across machines) half precision changes nothing that is stored.
    """
    if not device.startswith("cuda"):
        return {}
    import torch
    return {"model_kwargs": {"torch_dtype": torch.float16}}


def get_reranker(name: str = DEFAULT_RERANKER, device: str = "auto"):
    """Return (rerank, name) where rerank(query, docs) -> list[float].

    ``device="auto"`` = GPU first (cuda/rocm/xpu/mps), CPU only without a GPU.
    """
    from sentence_transformers import CrossEncoder

    from finetune_studio.smart_brain.devices import (
        encode_batch_size,
        resolve_device,
    )
    device = resolve_device(device)
    batch = encode_batch_size(device)

    # Strip known library prefixes that were incorrectly saved to manifests
    clean_name = name
    for library_prefix in ("sentence-transformers/shared:", "cross-encoder/shared:"):
        if clean_name.startswith(library_prefix):
            clean_name = clean_name[len(library_prefix):]
            break
    # Now resolve shared:reranker: references
    from finetune_studio.smart_brain.shared_refs import resolve_model_ref
    resolved = resolve_model_ref(clean_name, "reranker")

    local_path = None
    if resolved.startswith(RERANKER_LOCAL_PREFIX):
        local_path = resolved[len(RERANKER_LOCAL_PREFIX):]
    load_target = local_path if local_path else resolved

    model = CrossEncoder(load_target, device=device, max_length=512, **half_precision_kwargs(device))

    def rerank(query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        pairs = [[query, d] for d in docs]
        scores = model.predict(pairs, batch_size=batch, show_progress_bar=False)
        return [float(s) for s in scores]
    return rerank, name
