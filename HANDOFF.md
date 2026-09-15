# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 Europe/Warsaw)
| Area | Status |
|------|--------|
| GGUF export | Real llama.cpp path; success only with non-empty `.gguf` |
| GPTQ export | Prefers **`gptqmodel`**, falls back to `auto_gptq`; gated via `is_gptq_available()`; artifacts verified |
| fan-dragon tools | `gptqmodel` **installed** (7.5.0); `auto_gptq` absent; `~/llama.cpp` present |
| Artifact checks | Run `71002412`: GGUF F16 12.5 GB, Q4_K_M 2.50 GB, Q5_K_M 2.89 GB; GPTQ 2.67 GB, 4-bit/group-128, gptqmodel 7.5.0 |
| Artifact suites | Q4: GSM8K 83.3%, HellaSwag 100%, MMLU 100%; F16: 66.7%, 100%, 100%; GPTQ blocked by missing `optimum` in service venv |
| Tests | Focused GPTQ/export suite green locally; docs copy updated |

## Next steps
1. Add/install `optimum` for GPTQ inference, then rerun all three suites: `uv pip install --python .venv/bin/python optimum`.
2. Persist artifact-specific benchmark records instead of only direct harness runs.
3. Browser: verify the updated model-load error copy after deploy.
4. Truth check: `ss -ltnp | grep 7860` pid cgroup contains `finetune-studio.service`.

## Commands
- Tests: `.venv/bin/python -m pytest tests/test_advanced_quant.py tests/test_gguf_convert.py tests/test_run_export.py tests/test_export_capabilities.py -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/training/advanced_quant.py src/finetune_studio/training/run_export.py src/finetune_studio/training/export_capabilities.py tests/test_advanced_quant.py`
- Install GPTQ: `uv pip install --python .venv/bin/python -e '.[gptq]'` (= `gptqmodel>=2.0.0`, plus `optimum` for Transformers inference)
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- None for GPTQ on fan-dragon once this patch is pulled (gptqmodel already present).
