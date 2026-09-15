# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 Europe/Warsaw)
| Area | Status |
|------|--------|
| Offline benches | Smoke (6×3) + substantive synthetic offline (48/40/40) fixtures; labeled synthetic/offline |
| Training-data eval | `POST /api/testing/evaluate-training` + bench `…/evaluate-training`; leakage meta + per-case table |
| GPTQ deps | `.[gptq]` = `gptqmodel` + `optimum`; export UI shows “export only” when optimum missing |
| UI honesty | Unjudged pass_rate shows —; idle train badge not amber; GPTQ HF-inference hint |
| Tests | Focused offline / training-eval / GPTQ / badge suites (run locally) |

## Next steps
1. Pull on fan-dragon; install GPTQ+optimum: `uv pip install --python .venv/bin/python -e '.[gptq]'`
2. Restart: `systemctl --user restart finetune-studio`
3. Rerun artifact suites including GPTQ after optimum is present
4. Browser: Testing → Train-set eval; Benchmarks → offline suites + Train-set eval
5. Truth check: `ss -ltnp | grep 7860` pid cgroup contains `finetune-studio.service`

## Commands
- Tests: `.venv/bin/python -m pytest tests/test_benchmarks_offline_suites.py tests/test_training_eval.py tests/test_advanced_quant.py tests/test_export_capabilities.py tests/test_status_badge_honesty.py tests/test_benchmarks_industry_smoke.py -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/benchmarks src/finetune_studio/testing/training_eval.py src/finetune_studio/training/advanced_quant.py src/finetune_studio/training/export_capabilities.py src/finetune_studio/webui/routes/testing.py src/finetune_studio/webui/routes/benchmarks.py`
- Install GPTQ+optimum: `uv pip install --python .venv/bin/python -e '.[gptq]'`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- None for code review/commit on genorbox1; GPTQ Transformers load on fan-dragon still needs `optimum` installed in the service venv after pull.
