# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 Europe/Warsaw)
| Area | Status |
|------|--------|
| GGUF export | **Real llama.cpp path** in `training/gguf_convert.py` (discover + convert + verify). Sync UI + async worker share it. Success only with non-empty `.gguf`. |
| GPTQ export | `auto_gptq` gated in UI/API; success requires verified config+weights via `verify_gptq_artifacts`. AWQ still removed. |
| fan-dragon tools | `~/llama.cpp` convert + `llama-quantize` present; `auto_gptq` **not** installed yet |
| Tests | Focused export suite green locally; changes **uncommitted** for parent review |

## Next steps
1. Review uncommitted export diff; then commit + `git push`.
2. Fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio`.
3. Browser: Export GGUF `q8_0` on a merged run — expect non-empty artifact + registry row.
4. Optional GPTQ: `uv pip install --python .venv/bin/python -e '.[gptq]'` then restart; Export GPTQ when checkbox enabled.
5. Truth check: `ss -ltnp | grep 7860` pid cgroup contains `finetune-studio.service`.

## Commands
- Tests: `.venv/bin/python -m pytest tests/test_gguf_convert.py tests/test_export.py tests/test_run_export.py tests/test_export_capabilities.py tests/test_awq_removed.py tests/test_export_response.py -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/training/gguf_convert.py src/finetune_studio/training/run_export.py src/finetune_studio/training/advanced_quant.py src/finetune_studio/webui/routes/exports.py tests/test_gguf_convert.py tests/test_export.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- GPTQ needs `auto-gptq` on fan-dragon (see `docs/DEPLOYMENT.md` optional section). GGUF ready with existing `~/llama.cpp`.
