# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| Header nav | Desktop tabs wrap without overlap/scrollport; mobile uses hamburger only |
| Project/export mobile | Table scroll/actions and idle Stop fixes present in `a0ce3bb` |
| Chat idle/history | Empty state is not presented as a live completed result |
| RAG | Rebuild/search dedupe fix present and tested |
| Copy actions | Models/project/export copy buttons use escaped data attributes + clipboard fallback |
| Tests | `840 passed, 3 warnings`; focused nav/UI `44 passed, 2 warnings` |
| Browser audit | Verified fan-dragon at 375px and 1200px; screenshots and real hamburger/copy clicks passed |

## Next steps
1. Finish the slow-route Data Prep editor/modal pass from `docs/UI-AUDIT-PLAN.md`.
2. Exercise RAG chunks/search/chat/download modals without confirming destructive actions.
3. Recheck Overview/Data Prep Preview behavior and remaining mobile action clusters.

## Commands
```
make test
.venv/bin/python -m pytest tests/test_header_nav.py tests/test_ui_reliability.py -q --tb=short
.venv/bin/ruff check tests/test_header_nav.py tests/test_ui_reliability.py
```

## Blockers
- No active blocker. GPU-dependent training/inference remains unverified on genorbox1 by design.
