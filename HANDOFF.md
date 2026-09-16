# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| Mobile tables/actions | Fixed: models + bench run tables use `.table-scroll` + rem floors; data toolbar wraps Move to folder |
| Inference ≤780px | Fixed: `.inference-layout` height auto on mobile; Load model actions stay in document flow |
| Settings log tail | Fixed: prefer `journalctl --user -u finetune-studio`; stale `/tmp/uvicorn.log` marked not live |
| Bench base w/ 0 runs | Fixed: suites branch shows base-model RUN + truthful empty hint (no longer gated on runs) |
| HF Esc modal | Fixed: `closeHfModal` + Escape + `role=dialog` / aria-modal / focus restore |
| Tests | 18/18 focused green; Ruff clean on changed Python |

## Next steps
1. Deploy when ready: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio` (do not deploy from this task).
2. Browser at 375px: `/models`, `/projects/{pid}/benchmarks`, `/projects/{pid}/data`, `/inference` — confirm actions reachable.
3. Settings log badge shows **live** (journal) not stale bare uvicorn PID.
4. Benchmarks with zero runs + configured `base_model`: base RUN visible.
5. HF Explorer View → Escape closes `#hf-modal`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_project_settings.py tests/test_benchmarks_template.py tests/test_ui_reliability.py::test_models_and_bench_mobile_table_scroll_contract tests/test_ui_reliability.py::test_data_toolbar_actions_wrap_on_mobile tests/test_ui_reliability.py::test_inference_load_button_reachable_on_mobile tests/test_ui_reliability.py::test_hf_modal_escape_close_contract -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/webui/routes/project_settings.py tests/test_project_settings.py tests/test_benchmarks_template.py tests/test_ui_reliability.py`

## Blockers
- None for this audit pass (browser confirm pending after deploy).
