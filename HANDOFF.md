# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 17:20 Europe/Warsaw)
| Area | Status |
|------|--------|
| Git | Helper defaults + Ruff cleanup uncommitted (no push yet) |
| Helper | 27B GGUF-only routing preserved (data-prep / agent chat) |
| AWQ | Reject path + registry/UI exclusion preserved |
| SSE | Activity/training/export live updates unchanged |
| Lint | Ruff clean on all parent-task touched Python files |
| Regression | 69 focused + 8 AWQ tests passed |

## Next steps
1. Review + commit: `git status` / `git diff` (include `src/finetune_studio/models/helper.py`).
2. Push and on fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio`.
3. Browser: data-prep page shows helper label; load `local-default` before Start prep.
4. Browser: `/projects/{pid}/testing` copy mentions helper for LLM-assisted suite gen.
5. Confirm `/api/providers` JSON has `helper_provider_id` / `helper_label`.
6. Install llama.cpp converter on fan-dragon before treating GGUF export as available.

## Commands
- Tests: `.venv/bin/python -m pytest tests/test_helper_defaults.py tests/test_helper_model.py tests/test_agent_chat_model_resolution.py tests/test_data_prep_chat.py tests/test_data_prep_start.py tests/test_live_updates.py tests/test_awq_removed.py -v --tb=short`
- Lint: `.venv/bin/python -m ruff check src/finetune_studio/data/prep/generator.py src/finetune_studio/data/prep/runner.py src/finetune_studio/db/connection.py src/finetune_studio/models/manager.py src/finetune_studio/models/providers.py src/finetune_studio/models/registry.py src/finetune_studio/models/helper.py src/finetune_studio/webui/routes/data_prep.py src/finetune_studio/webui/routes/data_prep_chat.py src/finetune_studio/webui/routes/pages.py src/finetune_studio/webui/routes/training.py tests/test_agent_chat_model_resolution.py tests/test_data_prep_chat.py tests/test_data_prep_start.py tests/test_helper_defaults.py tests/test_helper_model.py tests/test_live_updates.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`
- Service truth: `systemctl --user show -p MainPID --value finetune-studio` then `grep finetune-studio.service /proc/<pid>/cgroup`

## Blockers
- GGUF conversion still needs llama.cpp tooling on the GPU host; API fails honestly until present.
- GPTQ still needs `auto_gptq` where used; AWQ intentionally not advertised.
- Changes not committed/pushed — fan-dragon cannot see helper defaults until deploy.
