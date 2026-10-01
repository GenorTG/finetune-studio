# `training/` — fine-tuning engine, export, VRAM estimation, data guards

Runs the actual SFT/LoRA training loop in an isolated child process, merges
and exports the result (GGUF / abliterated / merged safetensors), estimates
GPU memory before a run starts, and hosts a family of data-quality /
knowledge-preservation utilities that are mostly reachable only via CLI
(`fts ...`) rather than the live training pipeline.

## `engine.py` — `TrainingEngine`, the live training loop

The process-wide singleton (owned by `webui/app.py`) that drives a run.
`TrainingConfig` (dataclass) holds every tunable (LoRA rank/alpha, LR,
epochs, `gguf_quants`, `abliterate`, `system_prompt_mode`, etc.) and is
spawn-picklable via `worker.config_to_dict`/`_config_from_dict`.
`TrainingState` is the live progress snapshot (`status`, `loss`,
`current_step`, `log_lines`, …) that `monitor.py` serializes for the UI.

- `TrainingEngine.start()` never trains in-process. It spawns a
  `multiprocessing` **spawn** context child running
  `worker.training_worker`, so uvicorn's process never imports `unsloth`
  (GH E2E-40/E2E-27 — forking a tokenizer pool inside a CUDA-initialised
  threaded server deadlocks). A `_worker_target` test hook swaps in a
  `threading.Thread` with the same Queue protocol for unit tests.
- `_listen_child()` drains the child's `multiprocessing.Queue` for
  `{"op": "state", ...}` / `{"op": "done"}` messages and mirrors them onto
  `self.state` via `_apply_state_dict`. If the child dies without a clean
  terminal status, it synthesizes `"stopped"` (user-requested) or
  `"error"` (nonzero exit code) so a crashed child is never silently
  reported as still running.
- `_train()` is the actual pipeline body, run **inside the child**:
  format data (`training.data.format_for_sft`) → `split_data` (10% holdout)
  → `_train_unsloth` (falls back to `_train_standard` on `ImportError`) →
  optional merge/export/abliteration/auto-suite → `status = "done"`.
  `_train_unsloth` and `_train_standard` are two full, independently
  maintained training-loop implementations (Unsloth `FastLanguageModel`
  vs. plain `transformers` + `peft.LoraConfig`) — **this is the single
  live Unsloth training path**; see the Gotchas section below for the
  duplicate that was removed.
- **GGUF/merge auto-wiring gotcha**: `export_gguf=True` converts
  `<output_dir>/merged/`, which only exists if `merge_on_save` ran first.
  Both `_train_unsloth` and `_train_standard` detect
  "`export_gguf` but no `merged/` yet" and auto-merge before exporting
  (comment cites real incidents: run `8a1962c3` and `6e9e2672` asked for
  `q8_0` GGUF and got silent no-ops before this guard existed).
- `_do_export_gguf(output_dir, force=False)` reads quants from
  `self.config.gguf_quants`, calls `gguf_convert.convert_merged_to_gguf`,
  and returns that dict unchanged (`ok`/`status`/`files`/`quants`/`error`
  keys) — see **run_export.py** below for why this exact shape matters.
- `_do_abliteration()` calls `abliteration.abliterate_model` on
  `<output_dir>/merged/` and returns its raw dict (includes a numpy
  `refusal_direction` array — callers must sanitize before JSON-encoding;
  `run_export.py` does this correctly by hand-picking fields).
- `_maybe_merge()` / merge failures are **non-fatal**: the adapter on disk
  is still valid even if the 16-bit merge fails, so the exception is
  caught, folded into `state.message`/`state.error`, and training still
  reports `"done"`. `_sync_run_error()` additionally best-effort persists
  that error string onto the run row so "done" never silently hides a
  broken post-train export.
- `merge_adapter_for_run(run, force=False)` is the **standalone** /
  already-trained-run merge path (used by the export flow), distinct from
  `_do_merge` (used right after training finishes, with the model still
  in memory). Both resolve the 16-bit base via `merge_base.resolve_merge_base`
  and are idempotent (`_merged_dir_complete` short-circuits a re-merge).

### Fixed during this audit
`_persist_run_output()` computed `run_id` as the **first statement inside**
its `try:` block. If the `from finetune_studio.db.runs import update_run`
line itself raised (e.g. transient import failure), `run_id` was never
bound, and the `except Exception: log.exception("...", run_id)` handler
then raised an unrelated `NameError` while formatting the log message —
masking the real failure and propagating uncaught out of
`_persist_run_output()`, which `_train()` calls with no surrounding
try/except. A successful training run could therefore be mis-reported as
`"error"` by `_train`'s outer handler purely because of a logging bug.
Fixed by computing `run_id` before the `try:` (engine.py:570-572).
Regression: `tests/test_training_module_audit.py::test_persist_run_output_survives_db_import_failure`.

## `worker.py` — spawn-child entry point

`training_worker(config_dict, training_data, system_prompt, out_queue, stop_event)`
is the target function passed to `multiprocessing.Process`. It rebuilds a
`TrainingConfig` from a plain dict (`_config_from_dict`, filtering to known
dataclass field names so stray keys from an older schema don't crash it),
wraps the multiprocessing `stop_event` in a `threading.Event`-compatible
`_StopBridge`, subscribes a `_push` callback that forwards every state
change to `out_queue`, and calls `TrainingEngine._train()`. Always pushes
`{"op": "done"}` in a `finally`, even on an unexpected crash, so the parent's
`_listen_child` loop never blocks forever on a dead child.
Sets `UNSLOTH_DATASET_NUM_PROC=0`, `TOKENIZERS_PARALLELISM=false`, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` **before any
datasets/unsloth import** — the alloc-conf setting specifically fixes an
OOM at step ~40 on a 24 GiB card from cuBLAS/pool fragmentation on long
rank-128 runs (incident `e3f1ae03`).

## `data.py` — SFT formatting

`format_for_sft(data, system_prompt="")` normalizes four possible input
shapes (`conversations` ShareGPT, `messages`, `text`, `prompt`+`completion`)
into `{"messages": [...]}`. When both `conversations` and `messages` are
present it **prefers `conversations`** — the data-prep export endpoint
writes both, and a row with both is treated as an export artefact, not
user intent. `_conversations_to_messages` **raises** on an unrecognized
ShareGPT `from` role so a bad row fails the whole example loudly (caught
by `format_for_sft`'s `except ValueError: continue`, so only that one row
is dropped, not silently included with a wrong role) instead of quietly
mapping to a nonsense role. `clean_answer_for_training` strips
`source:`/`filename:`-style provenance lines/parens from assistant
responses before training (kept in the JSONL audit trail, not the label —
teaching the model filenames can turn digits into false numeric answers).
`split_data` does a seeded 90/10 shuffle-split (default `seed=42`) — this
is the "10%  validation holdout" referenced elsewhere in the codebase
(`testing/training_eval.py:build_heldout_eval`).

## `sft_args.py` — TRL config builder

`build_sft_training_args` / `build_sft_args_from_config` build an
`SFTConfig` (not `TrainingArguments`) specifically because TRL 0.24's
`SFTTrainer` does `dict_args.pop("push_to_hub_token")` on a converted
`TrainingArguments`, and transformers 5.x's `to_dict()` only emits
`hub_token` — the pop raises `KeyError` and training fails in ~8s before
any step runs. `push_to_hub=False` / `hub_token=None` so local training
never requires a HF token. Both `engine.py`'s two training paths and
`vram/profile.py`'s real-training VRAM profiler share this one builder —
no parallel SFTConfig construction exists elsewhere.

## `merge_base.py` — resolve a 16-bit base for QLoRA merge

QLoRA trains against a bitsandbytes/NF4-quantized base. Merging the
adapter back onto that quantized base and calling `save_pretrained` trips
transformers' `revert_weight_conversion` (`NotImplementedError`). The fix
is to find and load the **non-quantized sibling** of the training base in
bf16 and merge onto that instead.
`resolve_merge_base(base_model)`: if the given path's `config.json` isn't
quantized, return it unchanged. Otherwise strip quant suffixes
(`-unsloth-bnb-4bit`, `-bnb-4bit`, `-4bit`), derive `(org, bare_repo)`
from a HF hub cache path (`models--Org--Repo`), app download dir name
(`Org__Repo`), or `org/repo` Hub id, then search (in order)
`~/.cache/huggingface/hub` and `~/.finetune-studio/hf_models` for a
matching non-quantized snapshot, preferring `unsloth`/`Qwen`/the original
org. Raises `MergeBaseNotFound` with a concrete "download `Qwen/X`" hint
when nothing local matches — callers (`engine._do_merge`,
`engine.merge_adapter_for_run`) let this propagate as a user-facing error
rather than silently merging onto the wrong (quantized) base.

## `run_export.py` — sync post-training export

The one entry point external callers (`webui/routes/exports.py`, owned by
another lane) use to turn a finished run into a deployable artifact.
`export_trained_run(run, fmt="gguf", quants=None, force=False, base_model=None)`:
1. Rejects `awq`/`gptq` explicitly (both removed — unmaintained upstream
   packaging) and any format outside `{"gguf", "abliterated", "merged"}`.
2. `ensure_merged_for_export` merges the adapter if `merged/` doesn't
   already exist (via `engine.merge_adapter_for_run`), surfacing
   `MergeBaseNotFound` as a plain `ValueError` so the failure dict stays
   JSON-safe.
3. Dispatches by format: `merged` returns immediately;
   `abliterated` calls `engine._do_abliteration()` and hand-picks
   `refusal_magnitude`/`layers_modified`/`strength` into the response
   (never spreads the raw engine dict — it may carry a numpy array);
   `gguf` calls the private `_export_gguf` helper.

**Verified during this audit**: `_export_gguf`'s success/skip branches
return `"quants": quant_list` (the full requested list, case as given)
and `"files"` sourced from `verify_gguf_artifacts`'s `matched` list, which
is built by iterating `quant_list` **in order** and appending exactly one
hit per quant — `convert_merged_to_gguf` only reports `ok=True` when
*every* requested quant matched (no partial-success state exists). So on
every success/skip path, `files[i]` corresponds to `quants[i]`, same
length, same order — the invariant `webui/routes/exports.py`'s
one-DB-row-per-quant registration depends on. Pinned by
`tests/test_training_module_audit.py::test_convert_merged_to_gguf_files_quants_parallel`.
Only failure paths carry an incomplete `files` list, and those are always
paired with `ok=False`.

## `gguf_convert.py` — llama.cpp discovery + conversion (single source of truth)

Explicitly documented as "Used by sync export (`run_export`/`TrainingEngine`)
and the legacy async export worker" — i.e. this file is intentionally the
one shared implementation, not a duplicate.
- `find_gguf_convert_script()` / `find_llama_quantize()` search
  `$LLAMA_CPP_DIR`, `<repo>/.llama.cpp`, `~/llama.cpp`, `/opt/llama.cpp`,
  `/usr/local/llama.cpp` (in that order) plus a couple of legacy fallback
  paths and `PATH`.
- `convert_merged_to_gguf(merged_dir, gguf_dir, quants, force=False)`:
  normalizes quant labels, short-circuits with `status="skipped"` if a
  full matching artifact set already exists and `force` is false, and
  honors `FTS_SKIP_EXPORT=1` (writes 1-byte marker files instead of
  invoking llama.cpp — the standard test/CI short-circuit, checked
  *before* the merged-dir existence check so worker unit tests can use
  empty fixtures). For real conversion: single-step quants (`f16`,
  `bf16`, `f32`, `q8_0`) go straight through `convert_hf_to_gguf.py
  --outtype`; everything else (`q4_k_m`, `q5_k_m`, …) requires an
  intermediate f16 GGUF plus `llama-quantize`. **Never returns `ok=True`
  unless every requested quant has a verified non-empty `.gguf` on disk**
  (`verify_gguf_artifacts` re-check after conversion) — this is the fix
  for the historical "HTTP 200 / empty dir" false-success bug the module
  docstring references.
- `verify_gguf_artifacts(gguf_dir, quants=None)` is the one place that
  decides "did this quant actually get produced" — matches by filename
  stem (`gguf_filename_matches_quant`), accepting `q4_k_m.gguf`,
  `model-q4_k_m.gguf`, or any `*-q4_k_m.gguf`/`*_q4_k_m.gguf` suffix form.

## `export_response.py` — JSON-safe export API shape

`ExportResult` (pydantic) is "the single place" raw engine/export dicts
get sanitized before going into a FastAPI response — because abliteration
and other GPU paths can attach numpy arrays/tensors to their raw result
dicts, and `jsonable_encoder` raises `TypeError` on those, turning a
*successful* merge into an HTTP 500. `sanitize_export_dict` recursively
drops anything that isn't JSON-primitive/list/dict/numpy-scalar-like
(`_to_jsonable`'s `_DROP` sentinel) and maps `output_dir` → `output_path`
when the latter is absent. `ExportResult.from_raw` additionally normalizes
`ok`/`status` ("failed" when `error` is set or `ok is False`, else
"exported") and coerces list fields (`files`, `quants`, `missing`,
`layers_modified`, `supported`) to real lists even if they arrived as
another iterable. `artifact_path()` picks the best single display path
for the UI/DB: `output_path` → `merged_path` → `gguf_path` → `path` →
`files[0]`.

## `export_capabilities.py` — host capability probe

`probe_export_capabilities()` → `ExportCapabilities(gguf, gguf_script, gguf_hint)`
just wraps `run_export.find_gguf_convert_script()` (itself a re-export
from `gguf_convert.py`) so the Export UI can disable the GGUF option with
a concrete install hint instead of advertising a format that can only
fail.

## `abliteration.py` — refusal-direction removal ("de-censoring")

`detect_refusal_direction(model, tokenizer, ...)` runs 10 refusal-trigger
prompts and 10 safe-control prompts through the model, takes the mean
hidden state (last 4 layers by default) per group, computes
`refusal_mean - safe_mean`, and takes the dominant SVD singular vector as
the "refusal direction". `abliterate_model(model_path, output_dir,
strength=1.0)` loads the model fresh, detects the direction, then
projects it out of `lm_head.weight` only (`W -= strength * (W @ r̂) r̂ᵀ`) —
a single-point modification, not a full-model edit. Saves via
`safetensors.save_file` on a **cloned** state dict (bypasses `save_pretrained`
specifically because Unsloth's weight conversions aren't reversible on
save) and copies `chat_template.jinja`/`system_prompt.txt` if present.
`test_refusals(model, tokenizer)` generates real completions for the same
10 refusal prompts and keyword-matches the response for refusal language
— used to measure before/after abliteration effectiveness.
Called from `engine._do_abliteration()` and `run_export.export_trained_run`
(format `"abliterated"`); **not** gated by any hallucination/knowledge
guard — the pipeline removes refusal behavior but makes no corresponding
data-quality check.

## `advanced_quant.py` — imatrix GGUF (**not wired into the live engine**)

`quantize_gguf_imatrix` / `generate_imatrix` implement importance-matrix
GGUF quantization against **hardcoded** llama.cpp paths
(`~/llama.cpp/convert.py`, `~/llama.cpp/quantize`, `~/llama.cpp/imatrix`) —
a narrower, independent path discovery from `gguf_convert.py`'s
configurable `llama_cpp_search_paths()`. It *is* reachable from the live
engine: `TrainingEngine._do_export_imatrix` calls
`quantize_gguf_imatrix` when `config.export_imatrix` is set, and
`webui/routes/training.py` exposes an endpoint for it. This is a second,
narrower GGUF path discovery mechanism living alongside `gguf_convert.py`
's — not a bug today (different feature: imatrix vs. plain quant), but a
candidate for consolidation onto `gguf_convert.llama_cpp_search_paths()`
if imatrix export ever needs the same configurability (`$LLAMA_CPP_DIR`,
project-local `.llama.cpp/`) that plain GGUF export already has.

## `monitor.py` — SSE progress feed

`training_snapshot(engine)` is the one place that turns `engine.state`
into the JSON dict the UI polls/streams (includes `log_lines[-40:]`, i.e.
only the last 40 lines are ever sent per snapshot — older lines are
retained on `state.log_lines` itself but never pushed over SSE).
`training_events(engine)` is an `async` SSE generator: polls every 0.5s,
only yields a real `sse_data` frame when a composite key (status, step,
loss, message, error, log line count, last log line) changes; otherwise
yields a keepalive `sse_comment()` so the connection survives idle
periods without a full 2-second redraw poll.

## `run_persistence.py` — DB binding for live runs

`make_run_state_persister(run_id, output_dir, update_run=None, clock=time.time)`
returns an `on_update` callback with its own closure state
(`started_logged`, `finished`). Once `finished` flips true (any terminal
status: done/error/stopped), **every subsequent call is a no-op** — this
is what prevents a later run's residual callback from ever rewriting an
earlier run's final status/loss/timestamp. `attach_run(engine, run_id,
output_dir, project_id=None, update_run=None)` unsubscribes any previous
persister before binding a new one, so the single process-wide
`TrainingEngine` only ever has one live persister at a time regardless of
how many runs have started in that process's lifetime.

## `preset_advisor.py` — tier-based hyperparameter proposals

`propose(tier, base_model_ref, dataset_path=None, pair_count_hint=None, ...)`
→ `Advisory`. Four tiers (`smoke`/`balanced`/`precision`/`overkill`) are
anchored to one empirical reference point documented in the module
docstring: a 515-pair factual dataset hit 95.1% strict recall at 772
optimizer steps (12 epochs × 515 pairs / effective batch 8) vs. only 67%
at 257 steps — i.e. **recall is governed by optimizer steps, not epoch
count**, so `propose` computes `epochs` from a `steps_floor` per tier and
raises epochs to clear that floor regardless of dataset size. Rank scales
down for small/sub-8B bases (`_rank_factor`) and LR scales down for large
bases (`_lr_for`); `guess_base_params_b` regex-parses `"27B"`/`"8.3B"`-style
tokens from a model name (requires the trailing `b`/`B` so GGUF quant
suffixes like `Q4_K_M` are never mistaken for a size). Used by
`webui/routes/training.py`'s preset endpoint; not called from the engine
itself.

## `config_optimizer.py` / `data_quality.py` / `data_augmentation.py` / `hallucination_guard.py` — CLI/API data tools, not wired into training itself

These four are pre/post-training **advisory** tools: they analyze or
augment a JSONL file sitting on disk, independent of any run. None of
them are called from `engine.py` or `worker.py` — training always trains
on exactly the data it's handed; nothing here gates, filters, or
auto-fixes a dataset before `_train()` runs.

- `data_quality.DataQualityAnalyzer().analyze(path)` runs 8 checks
  (format, duplicates via MD5 hash, role balance, length outliers,
  PL/EN language balance, system-prompt consistency, empty responses,
  hallucination-risk phrase frequency) and returns one `severity`
  (highest issue severity wins). `generate_fixes(analysis)` maps issue
  types to a suggested `fts ...` CLI command.
- `data_augmentation.DataAugmenter()` generates synthetic examples for 5
  weakness categories (`knowledge`, `refusal`, `language_balance`,
  `hallucination_guard`, `persona_preservation`) from small hardcoded
  template pools (10-20 Q/A pairs each) and samples **with replacement**
  (`random.choice` in a loop), so a requested `count` > the template pool
  size produces heavy duplication, not an error — by design, not a bug,
  but worth knowing before requesting large augmentation counts.
- `config_optimizer.TrainingConfigOptimizer().analyze_and_recommend(data, current_config)`
  returns rule-based `TrainingRecommendation`s (LR/epochs/rank/weight-decay)
  keyed off dataset size and two heuristics computed from message content:
  `_calculate_pl_ratio` (fraction of user turns with >5% Polish diacritics)
  and `_calculate_persona_ratio` (fraction of user turns containing a
  persona/work keyword).
- `hallucination_guard.HallucinationGuardrail().check_response(question, response)`
  pattern-matches a **single live model response** for confidence
  language, fabricated-looking dates/numbers/URLs/emails, and
  question/response word-overlap, returning a `GuardrailResult`. The
  separate `TrainingDataValidator().validate_dataset(data)` scans an
  entire **training file** for the same risky patterns and returns an
  aggregate risk ratio + a plain-English recommendation string.

All four are correctly reachable from the CLI (`cli/commands/analyze.py`,
`augment.py`, `optimize.py`, `validate_hallucination.py` — verified by
reading each; they use the real class names and real method signatures).

## `knowledge_preservation.py` — `KnowledgePreserver` (orphaned, zero callers)

Implements data-mixing (`data_mixing`, persona:general ratio blending),
a `replay_buffer` sampler, and `ewc_hint`/`progressive_unfreezing_hint`
(return descriptive dicts only — no actual EWC/unfreezing implementation,
just documentation-as-data for a UI that doesn't exist yet), plus its own
`generate_knowledge_data`/`generate_refusal_data`/`balance_dataset`
helpers that duplicate similar-but-not-identical logic already in
`data_augmentation.DataAugmenter`.

**Confirmed via full-repo grep: `KnowledgePreserver` has zero callers**
anywhere in `src/` or `tests/` — not from the CLI, not from any webui
route, not from `engine.py`/`worker.py`. It is not wired into the live
training flow at all (this directly answers Priority Check 3: unlike
`hallucination_guard.py`, which is at least reachable via CLI/the
`validate-hallucination` command, this module is completely unreachable
from any entry point).

**Not deleted in this pass** — unlike `unsloth_engine.py`, this is not a
duplicate of a live implementation; it's a materially different, complete
feature (data-mixing ratios, replay buffer) with no equivalent live code
to prefer. Deleting a whole unreferenced-but-correct feature module is a
product decision (wire it in, or intentionally drop it), not a
correctness fix, so it's flagged here rather than removed. See
"Cross-module findings" in the audit report for the recommended next
step.

## `vram_profiler.py` — intentional back-compat shim (not a duplicate)

```python
"""Back-compat shim — the real code lives in finetune_studio.training.vram/."""
from finetune_studio.training.vram import (...)
```

Confirmed via full-repo grep this is a deliberate compatibility re-export
(its own docstring says so), not an unintentional parallel implementation
of `training/vram/`. Both the CLI (`cli/commands/vram.py`) and a test
file (`tests/unit/test_vram_profiler.py`) still import through this
shim; `tests/test_vram.py` imports the real package paths directly. No
"one true way" violation — the actual formulas/data live in exactly one
place (`training/vram/`), this file only re-exports names for callers
written against the old flat-module layout.

## `training/vram/` — the real VRAM estimation package

- `constants.py`: `DTYPE_BYTES` (bytes/param per dtype), `CUDA_OVERHEAD_GB`
  (0.5, empirical RTX 3090), `ACTIVATION_SAFETY_MARGIN` (1.15×),
  `MODEL_PRESETS` (hidden size / layer count for ~13 well-known models,
  used by `recommend_for_model`/`report.py`'s comparison table).
- `schema.py`: plain dataclasses only — `GPUInfo`, `VRAMEstimate` (+
  `to_dict()`), `ProfileResult`, `RecommendedConfig`.
- `gpu.py`: `detect()` reads real CUDA device properties via
  `torch.cuda.get_device_properties`/`mem_get_info`; any exception
  (no CUDA, no torch) falls back to a zeroed "Unknown" `GPUInfo` rather
  than raising, so every caller can treat GPU detection as always
  succeeding.
- `estimate.py`: `estimate_vram(model_size_b, method, batch_size,
  seq_length, lora_rank, ...)` — the 4-component formula (weights +
  gradients + optimizer states + activations + CUDA overhead) described
  in the module docstring. QLoRA/LoRA estimate adapter-only
  gradients/optimizer via a rough `adapter_params ≈ num_params * min(rank
  / hidden, 0.05)` approximation; full fine-tune counts every param.
  `fits = headroom_gb > 0.5`.
- `recommend.py`: `recommend_config` brute-forces
  `method × batch_size × seq_length` combinations, keeps only ones that
  `fits`, scores them (full_ft > lora > qlora, then longer seq, then
  bigger batch) and returns the top 5. `recommend_for_model(model_name)`
  looks up `MODEL_PRESETS` by exact or partial key match.
- `profile.py`: `profile_training(...)` is the **only function in this
  package that actually trains** — runs a handful of real steps (default
  5) on a tiny synthetic dataset and reads `torch.cuda.max_memory_allocated()`
  for ground-truth VRAM, as a check against `estimate_vram`'s formula.
  `profile_all_sizes` is estimate-only (despite the "profile" name) —
  it calls `estimate_vram`, not `profile_training`, for each entry in a
  `{label: hf_path}` dict; the `path` is carried through the result but
  never actually loaded.
- `report.py`: `generate_vram_report()` renders a full Markdown report
  (GPU info + a comparison table across every `MODEL_PRESETS` entry +
  the formula + optimization tips) and optionally writes it to disk.

## Gotchas / invariants for future changes

1. **Unsloth has exactly one live implementation now.** `engine.py`'s
   `TrainingEngine._train_unsloth` is it. `training/unsloth_engine.py`
   (`train_with_unsloth`/`is_unsloth_available`) was a complete,
   independently-written duplicate with **zero callers anywhere in the
   repo** (confirmed by full-repo grep before deletion) — it even
   silently swallowed the `chat_template.jinja` save failure with a bare
   `except Exception: pass` where `engine.py`'s version logs a warning.
   It has been deleted in this pass; do not recreate a second Unsloth
   training path — extend `TrainingEngine._train_unsloth` instead.
2. **GGUF `files`/`quants` parity is load-bearing.** Any change to
   `gguf_convert.convert_merged_to_gguf` or `run_export._export_gguf`
   must preserve "success ⇒ `files[i]` matches `quants[i]`, same length,
   same order" — `webui/routes/exports.py` depends on this to register
   one DB row per quant. See `test_convert_merged_to_gguf_files_quants_parallel`.
3. **`export_gguf`/`merge_on_save` auto-merge guard.** If you add a new
   post-train export format, check whether it also needs merged weights
   and replicate the "auto-merge if `export_gguf` is set but `merge_on_save`
   wasn't" guard in both `_train_unsloth` and `_train_standard` — this
   exact bug (GGUF requested, merge unchecked, silent no-op) has bitten
   two real runs (`8a1962c3`, `6e9e2672`).
4. **GPU/CPU offload contract (GH-AAA, `models/llama_loader.py`):** this
   lane's training paths use `device_map={"": 0}` (full GPU) with a
   `device_map="auto"` (mixed RAM+VRAM) fallback only inside
   `_load_model_with_fallback` (the **standard**, non-Unsloth path) on
   an actual CUDA OOM. No new violation of the mixed-offload contract
   was found in this lane.
5. **`knowledge_preservation.py` is dead code, intentionally not
   removed.** See its section above — flag before deleting; it's a
   product call, not a bug fix.
