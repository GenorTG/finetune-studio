#!/usr/bin/env python3
"""Test InferenceEngine loading."""
import sys
sys.path.insert(0, "src")

from finetune_studio.testing.inference import InferenceEngine

engine = InferenceEngine()
engine.load("/home/genortg/inference-server/models/chris-ai-v20.Q4_K_M.gguf")
print(f"Loaded: {engine.model_path}, is_gguf: {engine.is_gguf}")

result = engine.generate([{"role": "user", "content": "What is 2+2?"}], max_tokens=50)
print(f"Response: {result}")

engine.unload()
