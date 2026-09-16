# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| `options_json` decode | Fixed: `row_to_dict` decodes `options_json` → `options` dict (same contract as `settings_json`) |
| `test_parsed_converts_txt` | Fixed: disable auto-promote in fixture so GET /parsed hits on-the-fly `converted` (txt upload otherwise writes `files/<sha12>/parsed.txt` → `sibling`) |
| Helper / bench copy tests | Fixed: assert current data-prep “load that provider first” + benchmarks “not industry” wording |
| AWQ registry assertion | Fixed: behavior check via `_safe_model_name` + `_TRAINING_EXCLUDED_FORMATS` (AWQ still training-excluded) |
| Six-failure suite | 6/6 green; focused related 65/65; Ruff clean on touched files |

## Next steps
1. Run full local suite: `.venv/bin/python -m pytest tests/ -v --tb=short`.
2. Deploy: `git push`; fan-dragon `cd /home/genortg/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio`.
3. Confirm `:7860` cgroup is `finetune-studio.service`: `ss -ltnp | grep 7860` then check `/proc/<pid>/cgroup`.
4. Repeat browser audit on remaining project routes at 375/780/1280px.
5. Verify live RAG chat citations after rebuild + new parsed source.

## Commands
- Six failures: `.venv/bin/python -m pytest tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options tests/test_file_library_apis.py::test_parsed_converts_txt tests/test_helper_defaults.py::test_data_prep_page_shows_helper tests/test_helper_model.py::test_data_prep_page_shows_helper_label tests/test_helper_model.py::test_registry_generic_dirs_exclude_awq tests/test_status_badge_honesty.py::test_benchmarks_recent_scores_show_judged_column -v --tb=short`
- Focused: `.venv/bin/python -m pytest tests/test_db_lifecycle.py::TestSystemUpdateLifecycle tests/test_update.py tests/test_file_library_apis.py tests/test_helper_defaults.py tests/test_helper_model.py tests/test_status_badge_honesty.py -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/db/connection.py tests/test_db_lifecycle.py tests/test_file_library_apis.py tests/test_helper_defaults.py tests/test_helper_model.py tests/test_status_badge_honesty.py`

## Blockers
- None for these six failures.
