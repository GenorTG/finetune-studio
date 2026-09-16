# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| Header nav | Desktop tabs wrap with no horizontal scrollbar; mobile uses hamburger only |
| Project/export mobile | Table scroll/actions and idle Stop fixes present in `a0ce3bb` |
| Chat idle/history | Empty state is not presented as a live completed result |
| RAG | Rebuild/search dedupe fix present and tested |
| Tests | `840 passed, 3 warnings`; focused nav/UI `44 passed` |
| Browser audit | Blocked: browser-control service disabled while gateway drains; no screenshot evidence yet |

## Next steps
1. Rerun the browser sweep from `docs/UI-AUDIT-PLAN.md` when browser control is available.
2. Verify desktop/mobile header overflow and hamburger links.
3. Verify `/models`, `/export`, Training, Chat and all project tabs at 375px.
4. Exercise Overview/Data Prep Preview and every modal/tab/action.
5. Fix only screenshot-confirmed visual defects; rerun focused and full tests.

## Commands
```
make test
.venv/bin/python -m pytest tests/test_header_nav.py tests/test_ui_reliability.py -q --tb=short
.venv/bin/ruff check tests/test_header_nav.py tests/test_ui_reliability.py
```

## Blockers
- Browser-control service is disabled during gateway drain. Do not restart the gateway without Master Genor's explicit approval.
