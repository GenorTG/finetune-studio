"""`fts train` — start a training run with a progress bar."""
from __future__ import annotations

import os
import sys
import time


def cmd_train(args) -> None:
    from finetune_studio.training.data import load_jsonl
    from finetune_studio.training.engine import TrainingConfig, TrainingEngine

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.data):
        print(f"Error: Data not found: {args.data}")
        sys.exit(1)

    data = load_jsonl(args.data)
    print(f"Loaded {len(data)} examples from {args.data}")

    config = TrainingConfig(
        model_path=args.model, output_dir=args.output,
        lora_rank=args.lora_rank, learning_rate=args.lr,
        num_epochs=args.epochs, batch_size=args.batch,
        max_seq_length=args.max_seq, unsloth=not args.no_unsloth,
    )

    engine = TrainingEngine()

    def on_progress(state):
        if state.status == "training":
            pct = (state.current_step / max(state.total_steps, 1)) * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            sys.stdout.write(f"\r[{bar}] {pct:.0f}% | Step {state.current_step}/{state.total_steps} | Loss: {state.loss} | ETA: {state.eta}s")
            sys.stdout.flush()
        elif state.status == "done":
            print(f"\n\nTraining complete! Output: {args.output}")
        elif state.status == "error":
            print(f"\n\nError: {state.error}")
        elif state.status in ("loading", "saving"):
            print(f"  {state.message}")

    engine.on_update(on_progress)
    engine.start(config, data, args.system_prompt)

    while engine.state.status in ("loading", "training", "saving"):
        time.sleep(1)

    sys.exit(0 if engine.state.status == "done" else 1)
