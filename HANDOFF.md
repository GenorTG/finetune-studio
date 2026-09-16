# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| Real benches | `real://mmlu|gsm8k|hellaswag` via `real_benchmarks.py`; `is_real_benchmark=true` + HF metadata; default sample 50, `full_run` / `num_samples` knobs |
| Synthetic benches | Unchanged smoke/offline; labels `synthetic · … (not industry)` |
| Scoring | Strict MCQ / GSM8K #### finals; no substring credit on real suites |
| Inference endpoint | `/api/chat-v2/inference/benchmark` returns nested results + scalar `overall` (no sum-of-dicts) |
| Tests | Real-bench suite + discovery/smoke/offline/template: 44 passed; Ruff clean on touched files |
| Deployment | Needs `git push` + fan-dragon pull/restart; first real run downloads HF datasets into `data/benchmarks/hf_cache` |

## Next steps
1. Deploy: `git push`; on fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`.
2. Smoke a bounded real suite on fan-dragon (`num_samples=20`) after cache warm.
3. Browser-check benchmarks page labels (`real ·` vs `synthetic ·`).
4. Fix two unrelated legacy failures (`test_db_lifecycle`, `test_parsed_converts_txt`), then `make test`.

## Commands
- Real + synthetic bench tests: `.venv/bin/python -m pytest tests/test_real_benchmarks.py tests/test_benchmarks_suite_discovery.py tests/test_benchmarks_industry_smoke.py tests/test_benchmarks_offline_suites.py tests/test_project_testing.py tests/test_strict_scoring.py tests/test_benchmarks_template.py -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/benchmarks/real_benchmarks.py src/finetune_studio/benchmarks/suite_defs.py src/finetune_studio/webui/routes/benchmarks.py src/finetune_studio/webui/routes/chat_v2.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- `tests/test_db_lifecycle.py::TestSystemUpdateLifecycle::test_create_with_options` expects a dict but receives the persisted JSON string.
- `tests/test_file_library_apis.py::test_parsed_converts_txt` sees a pre-existing sibling artifact and expects conversion.
