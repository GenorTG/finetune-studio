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
| Tests | `tests/test_inference_gptq_load.py` green locally (mocks); prior offline/badge suites unchanged |

## Next steps
1. Push from genorbox1: `git push`
2. fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only`
3. Ensure GPTQ package: `uv pip install --python .venv/bin/python -e '.[gptq]'`
4. Restart: `systemctl --user restart finetune-studio`
5. Truth check: `ss -ltnp | grep 7860` then confirm pid cgroup has `finetune-studio.service`
6. Smoke GPTQ chat/bench against `…/runs/<id>/gptq` (Torch backend; no Marlin JIT)

## Commands
- GPTQ infer tests: `.venv/bin/python -m pytest tests/test_inference_gptq_load.py tests/test_inference_unsloth_load.py -v --tb=short`
- Lint GPTQ load: `.venv/bin/ruff check src/finetune_studio/testing/gptq_load.py src/finetune_studio/testing/inference.py tests/test_inference_gptq_load.py`
- Install GPTQ: `uv pip install --python .venv/bin/python -e '.[gptq]'`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- None for this patch on genorbox1. Live GPTQ load still needs `gptqmodel` in the fan-dragon service venv.
