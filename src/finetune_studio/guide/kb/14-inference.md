---
id: inference
title: Inference page (load a model)
page: inference
keywords: inference load model context length n_ctx gpu layers offload vram unload kv cache memory estimate idle unload loaded
---
## Purpose
Load any model into memory, set how much context and how many layers go on the GPU, and chat with it. Only one model is resident at a time; loading another unloads the first.

## Workflow
1. Pick a model in the picker; the sliders show its layer count.
2. Set **Context length** (KV cache; larger uses more VRAM) and **GPU offload layers** (all layers on the GPU is fastest; the loader steps layers down automatically if memory is short, never shrinking context).
3. **Load model**; the memory bar estimates VRAM. **Unload** frees it.
4. Idle models auto-unload after 5 minutes by default (`FTS_IDLE_TIMEOUT`, 0 disables).

## Key controls
- `#model-select` — model picker.
- `#n-ctx` — context length slider (512–131072).
- `#n-gpu-layers` — GPU offload layers.
- `#load-btn` — load or reload the model.
- `#unload-btn` — unload and free memory.

## What to check
The header chip (`no model` or the model name) must match reality; GPU placement is reported honestly (layers that did not fit run on CPU).

## Common mistakes
Setting a huge context on a nearly full GPU; forgetting that training and helper jobs unload whatever else is loaded.
