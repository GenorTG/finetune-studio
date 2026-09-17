"""Shared serialization for the global inference engine.

The global ``inference_engine`` holds one model on one GPU. llama.cpp /
transformers generation is not safe to call concurrently on the same model
instance, so every blocking engine op (load / generate) must be serialized.

Route handlers are ``async``; running these blocking ops inline froze the
whole event loop (the activity SSE, status polls, navigation — the entire
WebUI — stalled for the full seconds-to-minutes of a load or generation).
The pattern is:

    from finetune_studio.webui.engine_guard import ENGINE_LOCK
    async with ENGINE_LOCK:
        result = await asyncio.to_thread(inference_engine.generate, ...)

``to_thread`` moves the blocking work off the loop (UI stays responsive);
``ENGINE_LOCK`` keeps concurrent loads/generations from racing on the same
model. A module-level ``asyncio.Lock`` binds to the running loop on first
use (Python 3.10+), so defining it here is safe.
"""
from __future__ import annotations

import asyncio

ENGINE_LOCK = asyncio.Lock()
