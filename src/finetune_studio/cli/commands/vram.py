"""`fts vram {report|check|profile}` — VRAM profiling & fit-check."""
from __future__ import annotations

import json
import sys

from finetune_studio.cli._vram_print import print_vram_check


def cmd_vram(args) -> None:
    from finetune_studio.training.vram_profiler import (
        MODEL_PRESETS,
        estimate_vram,
        generate_vram_report,
        profile_training,
        recommend_for_model,
    )

    if args.vram_command is None:
        print("Usage: fts vram {report|check|profile}")
        sys.exit(1)

    if args.vram_command == "report":
        vram = getattr(args, 'vram', None)
        report = generate_vram_report(available_vram_gb=vram, output_path=args.output)
        print(report)
        if args.output:
            print(f"\nSaved to {args.output}")

    elif args.vram_command == "check":
        vram = getattr(args, 'vram', None)
        model = args.model.lower().replace(" ", "-")

        # Try to find in presets
        matched_preset = None
        for key in MODEL_PRESETS:
            if model in key or key in model:
                matched_preset = key
                break

        # If no preset match, try parsing as a number (e.g. "7b" or "7")
        if matched_preset is None:
            try:
                size_b = float(model.replace("b", ""))
                est = estimate_vram(
                    model_size_b=size_b, method=args.method,
                    batch_size=args.batch, seq_length=args.seq,
                    lora_rank=args.lora_rank, available_vram_gb=vram,
                )
                if args.json:
                    print(json.dumps(est.to_dict(), indent=2))
                else:
                    print_vram_check(model, est)
                return
            except ValueError:
                pass

        if matched_preset:
            configs = recommend_for_model(matched_preset, available_vram_gb=vram)
            if args.json:
                out = [{
                    "method": c.method, "lora_rank": c.lora_rank,
                    "batch_size": c.batch_size, "gradient_accumulation": c.gradient_accumulation,
                    "max_seq_length": c.max_seq_length, "estimated_vram_gb": c.estimated_vram_gb,
                    "fits": c.fits, "notes": c.notes,
                } for c in configs]
                print(json.dumps(out, indent=2))
            else:
                print(f"\nRecommendations for {matched_preset} ({MODEL_PRESETS[matched_preset]['params_b']}B):")
                print(f"{'Method':<10} {'Rank':<6} {'Batch':<6} {'GradAcc':<8} {'Seq':<6} {'VRAM':<8} {'Fits':<6} Notes")
                print("-" * 80)
                for c in configs:
                    fit = "✅" if c.fits else "❌"
                    print(f"{c.method:<10} {c.lora_rank:<6} {c.batch_size:<6} {c.gradient_accumulation:<8} {c.max_seq_length:<6} {c.estimated_vram_gb:<8.1f} {fit:<6} {c.notes}")
        else:
            print(f"Unknown model: {args.model}")
            print(f"Known models: {', '.join(sorted(MODEL_PRESETS.keys()))}")
            print("Or use a number like '7b' or '14'")
            sys.exit(1)

    elif args.vram_command == "profile":
        print(f"Profiling {args.model} with {args.method} (batch={args.batch}, seq={args.seq}, rank={args.lora_rank})...")
        print("This will download the model if not cached. Using synthetic data.")
        print()

        result = profile_training(
            model_path=args.model, method=args.method,
            batch_size=args.batch, seq_length=args.seq,
            lora_rank=args.lora_rank, num_steps=args.steps,
        )

        if args.json:
            print(json.dumps({
                "model": result.model_name, "method": result.method,
                "batch_size": result.batch_size, "seq_length": result.seq_length,
                "lora_rank": result.lora_rank, "peak_vram_gb": result.peak_vram_gb,
                "measured_model_gb": result.measured_model_gb,
                "measured_adapters_gb": result.measured_adapters_gb,
                "training_steps": result.training_steps,
                "step_time_s": result.step_time_s,
                "success": result.success, "error": result.error,
            }, indent=2))
        else:
            if result.success:
                print(f"✅ Profile complete")
                print(f"   Peak VRAM:    {result.peak_vram_gb:.2f} GB")
                print(f"   Model load:   {result.measured_model_gb:.2f} GB")
                print(f"   Adapters:     {result.measured_adapters_gb:.2f} GB")
                print(f"   Step time:    {result.step_time_s:.2f}s")
                print(f"   Steps run:    {result.training_steps}")
            else:
                print(f"❌ Profile failed: {result.error}")
                sys.exit(1)
