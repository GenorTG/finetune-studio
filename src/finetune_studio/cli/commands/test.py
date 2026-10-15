"""`fts test` — load a model and chat with it interactively."""
from __future__ import annotations

import os
import sys


def cmd_test(args) -> None:
    from finetune_studio.inference import describe_worker, get_engine

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    engine = get_engine()
    print(f"Loading {args.model}...")
    engine.load(args.model)
    fmt = "GGUF" if engine.is_gguf else "safetensors"
    print(f"Model loaded ({fmt}). Type 'quit' to exit.\n")
    print(f"Served by {describe_worker(engine)}.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        response = engine.generate(
            [{"role": "user", "content": user_input}],
            max_tokens=args.max_tokens, temperature=args.temperature,
        )
        print(f"AI: {response}\n")

    engine.unload()
    print("Model unloaded.")
