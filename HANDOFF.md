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
| Browser audit | Verified fan-dragon at 375px and 1200px; screenshots and real hamburger/copy clicks passed |
| Deployment | fan-dragon `c9bc2c6`, `finetune-studio.service` owns :7860; debug API returns `release_channel=EARLY BETA` |
| Fan-dragon data reset | Clean slate completed: all project/data rows and generated app artifacts removed; service active |
| Retained models | Exactly Qwen3.8-27B GGUF + mmproj and Qwen3-4B HF weights remain discoverable |

## Next steps
1. Exercise one real 4B inference and one short training smoke test on fan-dragon; inspect activity current/previous task rows.
2. Run the browser walkthrough against the empty clean slate: `tests/run_qa.sh`.
3. Verify retained model discovery after any service change: `ssh fan-dragon 'curl -sSL http://127.0.0.1:7860/api/models'`.

## Commands
```
make test
.venv/bin/python -m pytest tests/test_header_nav.py tests/test_ui_reliability.py -q --tb=short
.venv/bin/ruff check tests/test_header_nav.py tests/test_ui_reliability.py
```

## Blockers
- Browser sweep verified at 375px/1200px: nav open/close, no overflow/overlap, retained model sizes, project-create cancel, tutorial skip, RAG search/chunks, Data Prep preview tabs, settings dialogs, and Chat idle state. Empty activity is truthful, but populated current/previous task history is unverified because the reset left no tasks. A real 4B inference/training run is still required for end-to-end proof; 27B execution is intentionally deferred due resource cost.
