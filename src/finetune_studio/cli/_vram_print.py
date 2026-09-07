"""Pretty-printer for VRAM check results (used by `vram check`)."""
from __future__ import annotations


def print_vram_check(label: str, est) -> None:
    """Pretty-print a VRAM check result.

    `est` is a `VRAMEstimate` from `finetune_studio.training.vram_profiler`.
    """
    fit = "✅ FITS" if est.fits else "❌ TOO LARGE"
    print(f"\n{label} — {est.method.upper()}")
    print(f"  Model weights:    {est.model_weights_gb:.2f} GB")
    print(f"  Gradients:        {est.gradients_gb:.2f} GB")
    print(f"  Optimizer:        {est.optimizer_states_gb:.2f} GB")
    print(f"  Activations:      {est.activations_gb:.2f} GB")
    print(f"  ─────────────────────────")
    print(f"  Total estimated:  {est.total_gb:.2f} GB")
    print(f"  Available:        {est.available_gb:.2f} GB")
    print(f"  Headroom:         {est.headroom_gb:.2f} GB")
    print(f"  Result:           {fit}")
    if est.fits:
        print(f"  Safe batch size:  {est.max_batch_size}")
        print(f"  Safe seq length:  {est.max_seq_length}")
