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
| Tests | Focused GPTQ/export suite green locally; **uncommitted** patch for review |

## Next steps
1. Review uncommitted GPTQ/`gptqmodel` diff; then commit + `git push`.
2. Fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio`.
3. Browser: Export GPTQ on a merged run — expect enabled checkbox + verified `gptq/` artifacts.
4. Optional legacy: `uv pip install --python .venv/bin/python 'auto-gptq>=0.7.0'` still works as fallback.
5. Truth check: `ss -ltnp | grep 7860` pid cgroup contains `finetune-studio.service`.

## Commands
- Tests: `.venv/bin/python -m pytest tests/test_advanced_quant.py tests/test_gguf_convert.py tests/test_run_export.py tests/test_export_capabilities.py -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/training/advanced_quant.py src/finetune_studio/training/run_export.py src/finetune_studio/training/export_capabilities.py tests/test_advanced_quant.py`
- Install GPTQ: `uv pip install --python .venv/bin/python -e '.[gptq]'` (= `gptqmodel>=2.0.0`)
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- None for GPTQ on fan-dragon once this patch is pulled (gptqmodel already present).
