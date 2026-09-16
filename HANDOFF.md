# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| Release label | Product is explicitly labeled **EARLY BETA**; semantic version remains `v0.1.0` |
| Header nav | Desktop tabs wrap without overlap/scrollport; mobile uses hamburger only |
| Project/export mobile | Table scroll/actions and idle Stop fixes present in `a0ce3bb` |
| Chat idle/history | Empty state states that a model must be loaded before sending |
| RAG | Rebuild/search dedupe fix present and tested |
| Copy actions | Models/project/export copy buttons use escaped data attributes + clipboard fallback |
| Tests | Full suite pending after `2ce52a0`; focused chat/editor `7 passed`; changed Python routes Ruff clean |
| Browser audit | Verified fan-dragon at 375px and 1200px; screenshots and real hamburger/copy clicks passed |
| Deployment | fan-dragon `2ce52a0`, `finetune-studio.service` owns :7860; debug API returns `release_channel=EARLY BETA` |

## Next steps
1. Re-run the full regression suite after UI copy fixes: `.venv/bin/python -m pytest tests/ -q --tb=short`.
2. Run the browser walkthrough against fan-dragon for any newly added routes: `tests/run_qa.sh`.
3. GPU training/inference remain environment checks, not genorbox1 checks: `ssh fan-dragon 'systemctl --user status finetune-studio'`.

## Commands
```
make test
.venv/bin/python -m pytest tests/test_header_nav.py tests/test_ui_reliability.py -q --tb=short
.venv/bin/ruff check tests/test_header_nav.py tests/test_ui_reliability.py
```

## Blockers
- Browser sweep is verified for nav, RAG search/chunks, Data Prep preview tabs, settings dialogs, and Chat idle state. Destructive actions were only opened and cancelled; GPU training/inference were not started.
