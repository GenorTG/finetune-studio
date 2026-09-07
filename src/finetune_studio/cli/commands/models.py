"""`fts models` — list discovered models."""
from __future__ import annotations

import json


def cmd_models(args) -> None:
    from finetune_studio.config import settings
    from finetune_studio.models.registry import scan_models
    dirs = settings.model_dirs + (args.dirs or [])
    models = scan_models(dirs)
    if args.json:
        print(json.dumps([{
            "name": m.name, "path": m.path, "format": m.format,
            "size_gb": m.size_gb, "architecture": m.architecture,
        } for m in models], indent=2))
        return
    if not models:
        print("No models found.")
        return
    print(f"{'Name':<50} {'Format':<12} {'Size':<10} {'Arch'}")
    print("-" * 90)
    for m in models:
        print(f"{m.name:<50} {m.format:<12} {m.size_gb:<10} {m.architecture}")
