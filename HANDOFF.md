# HANDOFF — finetune-studio

Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## Mission
Ship held-out quality + WebUI evidence that an outside reviewer can't dispute:
real root-cause fixes, real fan-dragon verification, no fake greens. Every
audit row is reproducible from raw artifacts in `.tmp/evidence-run/` and
the project's per-source qa/pairs/*.json.

## State (verified 2026-09-17 Europe/Warsaw)
| Area | Status |
|------|--------|
| Release | EARLY BETA v0.1.0; service active under `finetune-studio.service` |
| Clean slate | `/api/projects=[]` baseline reset; one retained project `fbcf7083` (Helios Fulfillment Evidence Run) |
| Retained models | Qwen3.8-27B GGUF + mmproj (helper), Qwen3-4B Transformers (base). Trained exports: `8587cee6` (quality-v2), `8b1dd006` (quality-v3-augmented), and `f4f627af` (source-disjoint quality-v4) |
| Held-out suite | 23 cases, deterministic 90/10 split of 246-pair dataset (seed=42) at `/home/genortg/.finetune-studio/projects/fbcf7083/held-out.json` |
| Training/eval contract | UI distinguishes held-out quality from memorization; run-suite responses retain raw transcript, judge input, scoring method, validity, errors, and provenance |
| Fidelity audit | `/api/projects/{pid}/data-prep/audit` checks raw hash, deterministic reparse, chunks, token coverage, pair provenance, chunk coverage, and export count. Benchmark audit (`/api/projects/{pid}/benchmarks/{bid}/audit`) independently recomputes every verdict via `recompute_cases`. |
| Quality-v2 baseline | 230 rows, run `8587cee6`, 156 optimizer steps / 6 epochs, merged + Q8 exported. Held-out: **1 pass / 7 partial / 15 fail = 19.6% weighted, 4.3% strict pass**. |
| Quality-v3 augmented | 230 → 246 rows via `scripts/augment_dataset.py` (16 source-grounded pairs covering held-out gaps: RK-04 owner, Oct 5 review, 2% rejection, Nadiya Petrov, CR-77, Ada Smit/Elian Mertens, 612/74 returns, OCTOPUS-7741 external_api, Oriole Packaging dates, C-17 stop, customer reply, Exception glossary, temperature-sensitive, unload_regression_fixed date, 25-unit reason code, Pavel Novak). |
| Quality-v3 run | `8b1dd006` on Qwen3-4B / 168 optimizer steps / 6 epochs / final_loss=0.1281 / 231 s wall / LoRA r=64, alpha=128. Merged + Q4_K_M+Q5_K_M+Q8_0+F16 exported at `output/projects/fbcf7083/runs/quality-v3-augmented/`. |
| Quality-v3 held-out | Live persisted benchmark `ed1af82e` on the same deterministic 23-case suite: **4 pass / 7 partial / 12 fail = 32.6% weighted, 17.4% strict pass**. Benchmark audit after `4364886`: **23/23 transcripts valid, 0 disagreements**, strict `source_critical_facts` is now the only recomputed score. The earlier 6/6 result came from a different manual run path and is not used as the live result. |
| Source-disjoint quality gate | `scripts/build_source_disjoint_split.py` created **194 train pairs / 28 sources** and **52 held-out pairs / 7 unseen sources**. Run `f4f627af` trained 6 epochs (`final_loss=0.1273`); explicit Q8 benchmark `bb896ad9` is **0 pass / 12 partial / 40 fail = 11.5% weighted**, with **52/52 valid transcripts and 0 audit disagreements**. This is the honest generalization result: the model memorizes source facts and does not transfer to unseen sources. |
| Evidence corpus | `.tmp/evidence-run/held-out-q8-4f801bcb.json` (baseline transcript), `held-out-q8-augmented.json` (post-aug), `held-out-audit.json` + `held-out-audit-augmented.json` (independent verdicts), `augmented-pairs-v2.jsonl`, `held-out-suite.json`. |
| Live UI verification | Fan Dragon `e2e_ui_qa.py`: **70 passed / 0 failed**. Live Training API confirms run `8b1dd006` is `done`, 168/168 steps, `final_loss=0.1281`; `/api/training/runs/8b1dd006/exports` exposes merged, adapter, F16, Q8, Q5, and Q4 artifacts. |
| Last code | `4d752cd` persists the exact benchmark artifact path (migration + API); `32d8dd4` adds source-disjoint split generation/tests; `2518377` lets benchmarks select the export; `4364886` preserves source provenance; `fa84208` moves Agent inference off the event loop. |

## Next steps
1. **Fix the product quality gap** — 52-case source-disjoint evaluation is 0 strict pass; decide whether the product contract is fine-tuning-only memorization or add retrieval/context to answer unseen source facts.
2. **Use explicit benchmark targets** — benchmark `5ad2053d` proves Q8 quality-v3 is 6/6/11; benchmark records now include `scores.model_path`, so merged vs quantized exports cannot be conflated.
3. **Wire `recompute_cases` to the WebUI benchmark audit endpoint** so any saved benchmark row recomputes its verdicts on click, not only via local Python.
4. **Keep the live WebUI green** — run `8b1dd006` has final loss and all exports; browser QA remains 70/70.
5. **Browser upload path** — `browser upload` action's `paths` array still hits the MiniMax args-normalizer quirk. Drop the strict-typed UI hint or pre-encode the array; the API upload path works fine.

## Commands
- Tests: `make test` (= `.venv/bin/python -m pytest tests/ -v --tb=short`); focused on the changed module.
- Lint: `.venv/bin/python -m ruff check src/` (the `make lint` target hides failures with `|| true`).
- Run: `make run` (dev-only box).
- Deploy: `git push origin main` → on fan-dragon: `cd /home/genortg/finetune-studio && git fetch origin main && git checkout <sha> && systemctl --user restart finetune-studio.service`; truth check `ss -ltnp | grep :7860` shows new pid under `finetune-studio.service` cgroup.
- Augment + rebuild dataset: `python /home/genortg/finetune-studio/scripts/augment_dataset.py --project-id <pid>`.
- Held-out rerun: `python /home/genortg/finetune-studio/.venv/bin/python /tmp/run_held_aug.py` (or POST to `/api/testing/run-suite` with `suite_path` + `model_path` + `project_id`).
- Audit a benchmark: GET `/api/projects/{pid}/benchmarks/{bid}/audit` returns every transcript + `recompute_cases` disagreement list.

## Blockers
- Browser upload action's `paths` array still hits the MiniMax args-normalizer coercion quirk; use the API upload path instead. (No app-code fix possible without a wrapper change.)
- Local 27B Agent requests can exceed 180 seconds even at 2,048 tokens; the event-loop fix keeps health endpoints responsive, but the request must be quarantined rather than counted as generated data when it times out.
- Training can OOM even when `nvidia-smi` shows free VRAM — residual `python` processes (old inference workers, zombies) can hold 10+ GiB. Check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and kill stale pids before kicking a new run.
