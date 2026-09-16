# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| Project `/models` mobile | Fixed: `project_models.html` `#trained-exports-table-scroll` + rem floors (Copy / Open in inference) |
| Export trained exports | Same scroll pattern on `export_models.html` |
| Training Stop idle | `#stop-btn` disabled until ACTIVE status; enabled in `applyStatus` |
| Chat idle / stale success | No live restore of localStorage into `#chat-msgs`; labeled “Previous session (not a live result)” |
| `/models` filter pills | `#models-index-filters` wraps + scroll; added missing `.flex-wrap` utility |
| Tests | 9/9 focused green; Ruff clean on changed test Python (no src Python edits) |

## Next steps
1. Deploy when ready: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio` (do not deploy from this task).
2. Browser 375px: `/projects/{pid}/models`, `/projects/{pid}/export` — scroll + actions reachable.
3. Training idle: Stop disabled; while running: Stop enabled.
4. Chat empty project: empty-state tip only; no “Done — N factual Q&A…” in live pane.
5. `/models` filters: all pills usable (wrap/scroll), no silent clip.

## Commands
```
.venv/bin/python -m pytest \
  tests/test_ui_reliability.py::test_project_and_export_trained_exports_mobile_scroll \
  tests/test_ui_reliability.py::test_models_index_filters_discoverable_on_mobile \
  tests/test_ui_reliability.py::test_training_stop_disabled_while_idle \
  tests/test_ui_reliability.py::test_training_start_disabled_until_dataset \
  tests/test_chat_v2_template_fixes.py -v --tb=short
# → 9 passed

.venv/bin/ruff check tests/test_ui_reliability.py tests/test_chat_v2_template_fixes.py
# → All checks passed!
```

## Blockers
- None for this pass (browser confirm pending after deploy).
