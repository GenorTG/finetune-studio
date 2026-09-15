# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 Europe/Warsaw)
| Area | Status |
|------|--------|
| GPTQ inference | Local GPTQ dirs load via `gptqmodel` `BACKEND.GPTQ_TORCH` (`testing/gptq_load.py`); tokenizer separate; generate/unload unchanged |
| GPTQ deps | `.[gptq]` = `gptqmodel` + `optimum`; export UI still shows optimum hint for HF path |
| Offline benches | Smoke + substantive synthetic offline fixtures; semantics untouched by GPTQ load fix |
| Tests | 39 benchmark/GPTQ/training/UI regression tests passed locally; changed-file Ruff clean |
| Live GPTQ | Run `71002412/gptq` loaded on fan-dragon with `gptqmodel 7.5.0` + `optimum`; service-owned :7860 |
| Live suites | GSM8K 39/40 (97.5%), HellaSwag 40/40 (100%), MMLU 46/48 (95.8%); 128/128 judged |
| Training eval | `d5035f10`, 6/6 (100%), labeled `training_leakage`; not a generalization score |

## Next steps
1. Keep benchmark scores labeled by suite size and judge method; do not call training leakage a generalization result.
2. If adding real held-out evaluation, provide a dataset with no training overlap and record its provenance.
3. Repo-wide Ruff debt remains outside this change; changed GPTQ files are clean.

## Commands
- GPTQ infer tests: `.venv/bin/python -m pytest tests/test_inference_gptq_load.py tests/test_inference_unsloth_load.py -v --tb=short`
- Benchmark/UI regression: `.venv/bin/python -m pytest tests/test_benchmarks_offline_suites.py tests/test_benchmarks_industry_smoke.py tests/test_training_eval.py tests/test_status_badge_honesty.py tests/test_testing_template.py tests/test_breakpoints_visual.py tests/test_dark_light_toggle.py tests/test_confirm_modal_markup.py tests/test_browser_result_surfaces.py tests/test_inference_gptq_load.py -v --tb=short`
- Lint GPTQ load: `.venv/bin/ruff check src/finetune_studio/testing/gptq_load.py src/finetune_studio/testing/inference.py tests/test_inference_gptq_load.py`
- Install GPTQ: `uv pip install --python .venv/bin/python -e '.[gptq]'`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- Repo-wide Ruff reports legacy findings outside the changed files; do not treat that as a GPTQ regression.
