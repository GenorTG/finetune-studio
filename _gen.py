#!/usr/bin/env python3
"""Generate all source files for Finetune Studio."""
import os

BASE = "/home/genorbox1/.openclaw/workspace/finetune-studio/src/finetune_studio"

def w(rel, content):
    p = os.path.join(BASE, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(content)

# training/data.py
w("training/data.py", open("/tmp/ft_data.py").read())
