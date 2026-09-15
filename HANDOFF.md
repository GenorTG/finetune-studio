# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 ~15:12 Europe/Warsaw)
| Area | Status |
|------|--------|
| Git | `17544f1` pushed; activity-state fix `32fe913` and training observability `cc57f94` are ancestors |
| Training startup | ✅ Browser-started Qwen3-4B run `38867d1b` reached `done` in 49.27s; settings show `merge_on_save: false` |
| Hub-token bug | ✅ `SFTConfig` fix prevents TRL 0.24 / transformers 5.5 `push_to_hub_token` KeyError |
| Browser | ✅ fan-dragon HTTP 200; service MainPID is in `finetune-studio.service` cgroup |
| Export UI | ✅ Browser confirms GGUF, GPTQ, AWQ; GGUF `q8_0` option visible |
| Model cleanup | ✅ Retained Qwen3-4B cache and local Qwen3.8-27B safetensors/GGUF; removed old 0.6B/Unsloth/other HF model caches and stale 0.6B project artifacts |
| Testing UI | ✅ Browser shows retained 4B/27B choices and a visible Run button; auxiliary embedder retained for RAG |
| Tests | ✅ `13 passed`; targeted Ruff checks clean. Broader legacy Ruff has pre-existing warnings in `engine.py` and related modules |

## Next steps
1. In browser, select the completed 4B run on Export and export safetensors/AWQ plus GGUF `q8_0`.
   `http://fan-dragon:7860/projects/04954e70/export`
2. Prepare/select the project test suite, then run it against each exported 4B format in Testing.
   `http://fan-dragon:7860/projects/04954e70/testing`
3. Verify per-case results, model output/log visibility, and unload/load transitions with screenshots.
4. If activity drawer has a live item, expand it and wait through one 2-second poll; confirm expansion persists.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_sft_args.py tests/test_training_start_defaults.py -q --tb=short`
- Targeted lint: `.venv/bin/python -m ruff check src/finetune_studio/training/sft_args.py tests/test_sft_args.py`
- Deploy: `git push`; on fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`
- Service truth check: `systemctl --user show -p MainPID --value finetune-studio; grep finetune-studio.service /proc/<pid>/cgroup`

## Blockers
- Export/test execution is not yet complete; no exported 4B variants or final per-format benchmark evidence yet.
- Activity persistence has focused coverage, but a live expanded-row browser check needs an activity item to exist.
