# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| Release label | Product is explicitly labeled **EARLY BETA**; semantic version remains `v0.1.0` |
| Header nav | Desktop tabs wrap without overlap/scrollport; mobile uses hamburger only; closed mobile menu is removed from hit-testing (`c9bc2c6`) |
| Project/export mobile | Table scroll/actions and idle Stop fixes present in `a0ce3bb` |
| Chat idle/history | Empty state states that a model must be loaded before sending |
| RAG | Rebuild/search dedupe fix present and tested |
| Copy actions | Models/project/export copy buttons use escaped data attributes + clipboard fallback |
| Tests | Full suite `840 passed, 3 warnings`; changed Python routes Ruff clean |
| Browser audit | Verified fan-dragon at 375px and 1200px; screenshots, nav, copy, modal, empty-state, and route checks passed |
| Runtime proof | Qwen3-4B loaded and answered `4`; bounded LoRA smoke completed 10/10 steps with loss `1.9445` then `0`; activity showed Training DONE and Inference LOADED |
| 27B runtime proof | Qwen3.8-27B GGUF loaded, answered `4` in `2.3s`, reported 15.7 GB weights + 4.06 GB KV cache, then unloaded successfully; training view correctly excludes it as GGUF inference-only |
| Activity correctness | Stale project-bound task rows are now omitted after project deletion; global model/download rows remain visible |
| Memory reporting | Inference memory estimate now includes live GPU usage/capacity when available instead of implying `used 0.0` |
| Cleanup | Disposable projects/datasets/runs/artifacts deleted; latest 27B training-probe project removed; inference model unloaded; projects and activity views are empty |
| Deployment | fan-dragon `c9bc2c6`, `finetune-studio.service` owns :7860; debug API returns `release_channel=EARLY BETA` |
| Fan-dragon data reset | Clean slate completed: all project/data rows and generated app artifacts removed; service active |
| Retained models | Exactly Qwen3.8-27B GGUF + mmproj and Qwen3-4B HF weights remain discoverable |

## Next steps
1. Run the browser walkthrough against the empty clean slate: `tests/run_qa.sh`.
2. Verify retained model discovery after any service change: `ssh fan-dragon 'curl -sSL http://127.0.0.1:7860/api/models'`.

## Commands
```
make test
.venv/bin/python -m pytest tests/test_header_nav.py tests/test_ui_reliability.py -q --tb=short
.venv/bin/ruff check tests/test_header_nav.py tests/test_ui_reliability.py
```

## Blockers
- 27B training cannot run from the retained GGUF asset: the Training view correctly lists only Transformers-compatible bases and says GGUF/GPTQ exports are inference-only. A Transformers-format 27B checkpoint would be required; the retained 4B Transformers model is the supported training smoke-test base. GPU failure/recovery paths remain untested.
