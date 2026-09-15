# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 Europe/Warsaw)
| Area | Status |
|------|--------|
| Git | Browser-defect fixes uncommitted (no push yet) |
| Data prep | `.txt/.md/.markdown/.log` uploads auto-promote to parsed sources |
| HF Pull | job_id → activity drawer; registry refresh includes `~/.finetune-studio/hf_models` |
| Agent chat | Tool results as status/table; raw JSON under Debug |
| Breadcrumb | Per-page `breadcrumb_tab` + SPA swaps `#project-breadcrumb` |
| SSE copy | No user-facing "Updated every 2s"; activity shows Live |
| Regression | 41 focused tests passed; Ruff clean on new modules; Node syntax OK |

## Next steps
1. Review + commit: `git status` / `git diff`.
2. Push and on fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio`.
3. Browser: Data Prep upload a `.md` → Source file picker lists it with chunk_count.
4. Browser: HF Explorer Pull → activity drawer shows Download job (do not need a full model).
5. Browser: Agent chat `list_sources` with empty set → "No sources found" card, not raw JSON.
6. Browser: SPA-nav training/data-prep → breadcrumb tab label matches page.

## Commands
- Tests: `.venv/bin/python -m pytest tests/test_upload_auto_promote.py tests/test_hf_pull_activity.py tests/test_agent_chat_tool_render.py tests/test_breadcrumb_page_label.py tests/test_data_prep_promote.py tests/test_live_updates.py tests/test_breadcrumb.py -v --tb=short`
- Lint: `.venv/bin/python -m ruff check src/finetune_studio/data/fs/qa.py src/finetune_studio/webui/routes/activity.py src/finetune_studio/webui/routes/hf_models.py src/finetune_studio/config.py tests/test_upload_auto_promote.py tests/test_hf_pull_activity.py tests/test_agent_chat_tool_render.py tests/test_breadcrumb_page_label.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- GGUF conversion still needs llama.cpp tooling on the GPU host.
- Changes not committed/pushed — fan-dragon cannot see these fixes until deploy.
