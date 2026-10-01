# Testing engine (`src/finetune_studio/testing/`)

Local inference wrapper plus the full benchmark-suite stack: build suites
from training data or approved QA, run a model against them, judge the
answers (heuristic / strict / AI / local-model), and aggregate scores. Shared
by the webui testing/benchmarks routes, the CLI, and `benchmarks/`.

## `inference.py` — `InferenceEngine`

The shared inference wrapper. As of 2026-10-01 this is the **same object**
shared between `webui/app.py`'s global `inference_engine` and
`ModelManager.engine` (verified live: `inference_engine is
get_manager().engine` → `True`) — there is exactly one loaded-model instance
in the process, not two engines that can drift out of sync.

- `load(model_path, ...)` dispatches on file type: `.gguf` → `_load_gguf()`,
  anything else (a safetensors directory) → `_load_hf()`. Always calls
  `self.unload()` first, so loading a new model always fully releases the
  previous one (VRAM/RAM) before building the new one.
- `_load_gguf()` **delegates entirely to `models.llama_loader.load_llama_gguf()`**
  — confirmed by reading both files: there is no local `Llama(...)`
  construction left in this file. This was the historical duplication problem
  described in `llama_loader.py`'s own module docstring (`LocalGGUFProvider`
  and `InferenceEngine` each built their own `Llama()` and drifted); it is
  gone now. `_load_gguf()` only stores the result (`result.llama`,
  `result.vision`, `result.mmproj_path`, `result.final_n_ctx`) and separately
  extracts the GGUF's chat template via `templates.renderer.extract_template_from_gguf`
  (template extraction is engine-level concern, not loader concern, so it
  correctly stays here rather than in `llama_loader.py`).
- `_load_hf()` loads safetensors checkpoints with plain `transformers`
  (never Unsloth — Unsloth monkey-patches `transformers` globally and would
  poison every later inference call in the same process; see
  `test_server_never_imports_unsloth.py`). Load order on OOM: full bf16 →
  bitsandbytes 4-bit (single GPU, `device_map={"": 0}`) → if even 4-bit OOMs,
  a last-resort `device_map="auto"` (HF's own CPU/GPU layer splitter). That
  last fallback **is not covered by the GH-AAA "no mixed CPU/GPU offload"
  contract** — GH-AAA (see `models/llama_loader.py`) is explicitly scoped to
  the GGUF/llama.cpp loader (its docstring: "model layers always stay fully
  on GPU; only the KV cache shrinks"). `training/engine.py` documents the
  exact same `device_map="auto"` last-resort pattern for HF loads elsewhere
  in the codebase, so this is consistent existing practice, not a new
  violation.
- `generate()` is the single entry point both load paths funnel through; it
  dispatches to `_generate_gguf()` (llama.cpp, respects `stop=`) or
  `_generate_hf()` (plain `transformers.generate()`).
- **Fixed in this audit**: `_generate_hf()` accepted a `stop` parameter but
  never used it — stop sequences worked for GGUF models and were silently
  dropped for every HF-loaded model. Now truncates the decoded response at
  the earliest matching stop string, mirroring llama.cpp's behavior. See
  `tests/test_testing_engine_module_audit.py::test_generate_hf_honors_stop_sequence`.
  No caller in the repo currently passes a non-`None` `stop` to `generate()`
  (grepped), so this was a dormant correctness bug in the public contract
  rather than an active one — still worth fixing since `generate()` is the
  shared interface every caller (chat, RAG, testing, benchmarks) can rely on.
- `_start_idle_timer()` / `_auto_unload()`: auto-unloads after
  `FTS_IDLE_TIMEOUT` seconds (default 1800) of inactivity via a daemon
  `threading.Timer`, so a forgotten loaded model doesn't hold VRAM forever.
- `unload()` is intentionally aggressive: drops all references, explicit
  `del` + `gc.collect()` on the calling thread (not relying on eventual GC)
  so `Llama.__del__` runs deterministically, then `torch.cuda.empty_cache()`.
  Comment in the code explains this was previously a latent VRAM-leak bug
  (model appeared "still loaded" in nvidia-smi for seconds-to-minutes after
  `unload()` returned).
- `estimate_memory()` / `read_model_metadata()`: GGUF VRAM/RAM estimates from
  a **hand-rolled GGUF header parser** (reads only the first few KB, no full
  model load) with a fallback to the `gguf` package's `GGUFReader` if the
  fast path can't find all four fields. Results are cached in
  `_GGUF_META_CACHE` keyed on `(path, mtime, size)`, capped at 32 entries
  (LRU via `OrderedDict.popitem(last=False)`).

**Gotcha**: `n_ctx` / `n_gpu_layers` on the instance reflect what the loader
actually built the model with, not what the caller asked for — OOM-retry in
`load_llama_gguf()` can shrink `n_ctx` below the request, and callers that
need the truth (e.g. a `describe()` endpoint) must read `engine.n_ctx`, not
the original request parameter.

## `suite.py` — core data model + suite execution

The foundational module every other file in this package (and most of
`benchmarks/`) imports.

- `BenchmarkCase`: question + known-correct answer + optional `keywords`,
  `source_id`/`chunk_idx` (for RAG grounding), `row_index` (full-coverage
  audit trail back to the source dataset row).
- `CaseResult`: one case's run+judge outcome — transcript, verdict,
  `scoring_method`, `validity`, `error`.
- `load_test_suite(path)`: loads a suite JSON, accepting either a bare case
  array or a `{"cases": [...]}` wrapper object. Supports a v1 fallback format
  (`messages` + `expected_keywords`) for legacy suite files alongside the
  current v2 format (`question`/`correct_answer`).
- `run_suite(engine, cases, ...)`: runs each case through `engine.generate()`,
  catches `Exception` per-case (so one bad case doesn't abort the whole
  suite) and records it as `CaseResult(model_answer="", error=str(e))`. No
  judging happens here — purely execution.
- `apply_heuristic_judging(results)`: the judging dispatcher. For each
  unjudged result (`not r.verdict`), in priority order:
  1. Skip entirely if already judged (`r.verdict` truthy) or if it errored
     with no answer (`r.error and not r.model_answer`) — **this second skip
     is why the `score_results()` denominator fix below mattered**: skipping
     here leaves `r.verdict == ""`, and `score_results()` used to exclude
     that case from its `pass_rate` denominator instead of counting it
     against the score.
  2. If `source_id` is set, try `strict_scoring.score_source_grounded()`
     (critical-fact + content-term coverage, used for RAG/source-grounded
     cases).
  3. Otherwise try `strict_scoring.score_strict()` (MCQ/numeric exact
     matching; returns `None` for open-ended questions, falling through).
  4. If `keywords` is non-empty, substring keyword matching (pass if all
     keywords hit, partial if some, fail if none).
  5. Last resort: `judge.judge_case_heuristic()` (key-word overlap against
     `correct_answer`, always returns a verdict unless `correct_answer` is
     empty — see `judge.py` below).
- `score_results(results)`: aggregates `passed`/`partial`/`failed`/`unjudged`
  counts and computes `pass_rate` and `weighted_score`.
  **Fixed in this audit** (Priority check 1): both were previously computed
  as `x / n_judged` (cases with a non-empty verdict only). A case that
  errored during execution (`run_suite`'s except branch) or whose judging
  itself failed (AI judge API error, local-judge exception, no judge API key
  configured — see `judge.py`) ends up with `verdict == ""` and was
  **silently excluded from the denominator**, which could inflate the
  reported pass rate arbitrarily (e.g. 5 pass / 5 errored out of 10 total
  used to report 100%, not 50%). `webui/routes/benchmarks.py` (another
  lane's module) copies `pass_rate` directly into `scores["accuracy"]`
  (`benchmarks.py:258`), so this bug would have reached the user-facing
  accuracy number. Now both metrics divide by `total` (all cases), so
  errored/unjudged cases count as 0 toward the score while `unjudged` stays
  available separately for diagnosing *why* the score is low. Pinned by
  `tests/test_testing_engine_module_audit.py::test_score_results_does_not_inflate_pass_rate_with_errored_cases`.
  When every case gets a verdict (`judged == total`), behavior is unchanged
  — confirmed no regression across the full existing scoring test suite (48
  tests in `test_full_corpus_suite.py`, `test_rag_suite.py`,
  `test_bench_judge.py`, `test_suite_full_coverage.py`,
  `test_training_eval.py`, `test_benchmarks_industry_smoke.py`,
  `test_benchmarks_run_validation.py`, `test_strict_scoring.py`,
  `test_benchmarks_offline_suites.py`).

## `judge.py` — judging strategies

Three independent judges, all returning `(verdict, reasoning, confidence)`:

- `judge_case_heuristic()`: no API key, no model — content-word overlap
  ratio between `correct_answer` and `model_answer` (`_PASS_RATIO=0.6`,
  `_PARTIAL_RATIO=0.3`). Always returns a verdict *unless* `correct_answer`
  has no extractable key words at all (returns `("", "no correct_answer to
  compare against", 0.0)`) — an intentional "can't judge this" signal, not a
  silent drop, and it now correctly counts against `score_results()`'s
  denominator per the fix above.
- `judge_case_ai(...)`: calls an OpenAI-compatible `/chat/completions`
  endpoint (`FTS_JUDGE_MODEL`/`FTS_JUDGE_API`/`FTS_JUDGE_API_KEY` env vars)
  with a structured JSON-response prompt (`JUDGE_PROMPT`: ignore style/length,
  only check facts). Returns `("", "no API key configured...", 0.0)` if no
  key, or `("", f"AI judge error: {e}", 0.0)` on any exception (network,
  malformed JSON, etc). Both are legitimate "judging failed" outcomes that
  flow into the same `score_results()` denominator fix.
- `judge_case_local(engine, ...)`: same prompt, but sent through the local
  `InferenceEngine` instead of an external API. Tries `json.loads()` on the
  raw response first; on `JSONDecodeError` falls back to crude substring
  sniffing (`"pass" in low and "partial" not in low` etc.) rather than
  giving up — a deliberate best-effort fallback, not a bug. A genuine
  exception (e.g. engine not loaded) returns `("", f"local judge error: {e}",
  0.0)`.

`build_judge_messages()` is the one shared prompt builder all AI/local judges
use — not duplicated between `judge_case_ai` and `judge_case_local`.

## `strict_scoring.py` — task-aware exact scoring

Replaces permissive keyword/substring matching for MCQ and numeric cases,
and does conservative critical-fact scoring for source-grounded (RAG) open
answers. Every code path through here returns a concrete `StrictScore`
(`verdict` always one of `pass`/`fail`/`partial`, never empty) — confirmed by
reading every return statement in `score_multiple_choice()`,
`score_numeric()`, and `score_source_grounded()`. This file never silently
drops a case; `score_strict()` only returns `None` as an explicit "not
MCQ/numeric, caller should use its own open-ended path" signal for
`detect_task_kind() == "open"`.

- `detect_task_kind()`: MCQ if the question has ≥2 lettered option lines or
  the correct answer is a bare letter with ≥1 option line present; numeric
  if the correct answer is a plain int/float; else open.
- `score_multiple_choice()`: requires an unambiguous single selected letter;
  multiple asserted letters (including the correct one plus a wrong one) are
  rejected as `wrong_extra`/`ambiguous`, not credited.
- `score_numeric()`: prefers a `####`-style final answer (GSM8K convention),
  then an explicit "answer is X" phrase, then the last bare number in the
  text; rejects cases where the expected number appears but isn't the final
  asserted value (treats it as the model "getting there but not committing").
- `score_source_grounded()`: extracts `extract_critical_facts()` (dates,
  times, IDs, percentages, numbers) and `extract_content_terms()` (lexical
  anchors) from both the correct answer and the model answer, requiring
  coverage of both; also checks named-entity presence for identity
  questions (`who`/`owner`/`contact`/etc.) and contradiction for
  approved/rejected-style facts. `_focused_expected_answer()` handles
  legacy table/record-shaped correct answers by picking only the row(s) that
  match question anchors, so a correct answer containing an entire
  unrelated table doesn't drag in facts the question never asked about.

## `generate_suite.py` vs `full_corpus_suite.py` vs `rag_suite.py` vs `suite.py` — four different jobs, not duplicates (Priority check 3)

These are **not** the same concern reimplemented four times. Two build
suites, two run them, from different inputs and in different modes:

| File | Role | Input | Output |
|---|---|---|---|
| `suite.py` | Core model + **plain** suite runner | n/a (library) | `run_suite()`: ask each case, no retrieval |
| `generate_suite.py` | Suite **builder** from training data | training JSONL (post-training) | writes `suite_<name>.json` to disk, full coverage by default |
| `full_corpus_suite.py` | Suite **builder** from approved QA | project's approved QA pairs (pre-training, via `data.fs.qa`/DB) | writes `full-ingested-corpus.json`, discoverable via `suite_defs` |
| `rag_suite.py` | **Retrieval-grounded** suite runner | a suite file (loaded via `suite.load_test_suite`) + a RAG corpus | retrieve → ground prompt → generate, same judging as `run_suite` |

`training_eval.py` is a fifth, narrower case: it builds `BenchmarkCase` lists
in-memory (no suite JSON file written) from training data for
leakage/held-out checks, and **reuses** `generate_suite.py`'s private
`_categorize`/`_analyze_difficulty`/`_slugify` helpers rather than
duplicating them — correct reuse, not a violation.

Why two builders and not one: `generate_suite.py` converts whatever was fed
to the trainer (JSONL on disk, no DB dependency — works from the CLI with
no server running). `full_corpus_suite.py` converts whatever's currently
*approved* in the QA review workflow (DB-backed `list_qa_pairs`, applies the
same `deduplicate_qa_pairs()` one-question/one-target contract as dataset
export) — these can diverge (a project's approved QA set is edited after
training, or a suite is generated from a JSONL that was never run through
approval). Why two runners: `run_suite()` asks the model directly;
`run_rag_suite()` forces retrieval-grounded answering (and includes retry
logic for a known actual-hours/variance table mix-up,
`_needs_table_arithmetic_retry()`) for suites specifically testing whether a
model can use retrieved context instead of (or in addition to) trained
knowledge. Confirmed: `rag_suite.py` imports and reuses `suite.py`'s
`BenchmarkCase`, `CaseResult`, `apply_heuristic_judging`, `load_test_suite`,
`score_results` directly rather than reimplementing any of them.

### `full_corpus_suite.py` — how it builds the "run against the whole corpus" suite

Backs the live `full-ingested-corpus` suite (verified 2026-10-01 at 91.1%
pass rate against a real trained model via `webui/routes/benchmarks.py`,
owned by another lane).

1. `cases_from_approved_pairs(project_id)` calls `data.fs.qa.list_qa_pairs()`
   for every QA pair in the project, filters to `dict` entries, then
   `_deduplicated_pairs()`:
   - keeps only `status == "approved"` pairs with a non-empty `source_id`
   - extracts `(question, answer)` via `extract_qa_text()` (checks
     `question`/`answer` fields first, falls back to `messages`/
     `conversations` role-based extraction)
   - runs the result through `data.prep.export.deduplicate_qa_pairs()` — the
     **same** dedup contract dataset export uses, so the suite can't contain
     duplicate question/target pairs that a real training export wouldn't.
2. Each surviving pair becomes a `FullCorpusCase` via `case_from_pair()`
   (re-validates status/source_id/Q&A non-empty — defensive, since
   `_deduplicated_pairs` already filtered, but keeps `case_from_pair()` safe
   to call standalone).
3. `_write_cases()` writes `build_suite_document(cases)` to
   `suite_path_for_project(project_id)` (`<project>/suites/full-ingested-corpus.json`).
   **Case count reconciles**: `case_count` in the written JSON
   (`build_suite_document`, line `"case_count": len(cases)`) is `len(cases)`
   from the exact same list that gets serialized into `"cases"` — not a
   separately tracked counter that could drift. No silent truncation: there
   is no `max_cases`/sampling parameter anywhere in this file's build path
   — every approved, deduplicated, valid pair becomes a case.
4. If there are zero usable cases, `_write_cases()` **deletes** a stale
   suite file if one exists and returns `written=False` — so an emptied-out
   project doesn't leave a stale, now-wrong suite file behind that discovery
   could still pick up.

**Fixed in this audit**: `cases_from_pairs_directory()` (the CLI/offline
`project_root/qa/pairs` directory-scan path, used by
`write_full_corpus_suite_from_project_dir`, not the live DB-backed path
above) previously swallowed unreadable/malformed pair files with a bare
`except (OSError, json.JSONDecodeError): continue` and no logging — a
malformed pair file would silently vanish from the suite with no trace. Now
logs a `WARNING` per skipped file (unreadable JSON or non-object content)
naming the file and the error, while still returning the cases it could
build from the rest. Pinned by
`tests/test_testing_engine_module_audit.py::test_cases_from_pairs_directory_logs_unreadable_files`.

## `generate_suite.py` — build a suite from training JSONL

`generate_suite_from_training_data()`: converts a training JSONL
(`conversations`-format rows) into a suite JSON on disk.

- **Coverage contract** (explicit in the docstring, dated 2026-09-20): the
  suite must cover **all** dataset rows by default — `max_cases=None` is
  full coverage; passing `max_cases` is an *explicit opt-in* to deterministic
  uniform sampling (`random.Random(sample_seed).sample(...)`), and when
  sampling kicks in the suite is renamed `<stem>-sampledKofN` and the
  returned dict carries `coverage="sampled"` so a sampled result can never be
  misread as full coverage by a caller that only checks `case_count`.
- Parses **all** rows first (tracking `invalid_lines` for JSON decode
  failures and `skipped` for rows missing a 2-message `conversations` array
  or empty question/answer), *then* samples — so a sampled suite is a
  uniform sample of the whole file, not just its head.
- Every count in the returned dict reconciles:
  `dataset_count` (total JSON lines parsed) → `pool_count` (valid cases
  before sampling) → `case_count` (final, possibly sampled, count) are all
  derived from the same lists at each stage, with `skipped`/`invalid_lines`
  as separate explicit counters — no silent gap between "N rows in" and "M
  cases out."
- `_categorize()`: keyword-based question-type classifier (code / numeric /
  factual / reasoning / process / temporal / spatial / comparison /
  knowledge / general) — first match wins, checked in a fixed priority
  order.
- `_analyze_difficulty()`: classifies by answer length only (`≤30` easy /
  `≤100` medium / else hard) and returns a `judge_hint` string describing how
  strictly that difficulty tier should be judged (exact match / key facts /
  semantic). This `judge_hint` is stored as `BenchmarkCase.context` — it's
  advisory text for an AI judge prompt, not consumed by any scorer in this
  lane's files (`strict_scoring.py` and `judge.py` don't read `context`).

## `rag_suite.py` — retrieval-grounded suite runner

- `RagSearchEngine` / `ChatEngine` are `Protocol`s (structural typing, no
  base class needed) — `run_rag_suite()` only needs `.search()`/
  `.format_context()` on the RAG side and `.generate()` on the chat side, so
  it works with `PortableRAG`'s real query engine, a test double, or in
  principle any compatible object.
- `run_rag_suite()` per case: `rag_query.search(question, top_k)` →
  if the case declares a `source_id` and none of the initial hits match it
  (`hit_matches_source()`), retries with a wider `top_k=max(20, top_k*2)`
  search and swaps in the expanded hits *only if* they contain a match (a
  "recovery pass" for short questions whose correct source gets crowded out
  by near-duplicate distractors in the first pass — the expanded search's
  extra noise is discarded once the right source is found, not merged in).
  Then grounds the prompt (`build_grounded_messages()` — forces a literal
  "I don't know" reply when the context lacks the answer, via
  `data.rag_eval.UNKNOWN_REPLY`) and generates. If
  `_needs_table_arithmetic_retry()` detects the common
  actual-hours/variance-column mixup in a tabular context, it re-prompts once
  with an explicit correction instead of accepting the first (likely wrong)
  answer.
- Exceptions during search/generate are caught per-case (mirrors
  `suite.run_suite()`'s pattern) and recorded as an errored `CaseResult` —
  these now also correctly count against `score_results()`'s denominator per
  the fix above, since `run_rag_suite_evaluation()` calls the same
  `apply_heuristic_judging()` + `score_results()` pipeline.
- `compute_retrieval_metrics()`: recall/hit-rate only over cases that
  declare a `source_id` (cases without one can't be scored for retrieval
  correctness, and are correctly excluded from that *specific* denominator —
  unlike the `pass_rate` bug, this exclusion is semantically required, not a
  reporting bug, since "did we retrieve the right source" is undefined for a
  case with no expected source).

## `training_eval.py` — training-data leakage / held-out evaluation (Priority check 2)

Builds `BenchmarkCase` lists directly from a project's training JSONL for
two distinct purposes, both explicitly labeled to prevent a high score being
misread as "the model generalizes":

- `build_training_eval()`: full/capped training-set recall check
  (`LEAKAGE_WARNING`: "High pass rates measure in-distribution recall /
  memorization risk, not generalization").
- `build_heldout_eval()`: the deterministic 10% slice the trainer itself
  holds out (`random.Random(42).shuffle(...)`, same seed as training —
  `split = int(len(shuffled) * 0.9)`, takes `shuffled[split:split+max_cases]`)
  — labeled as "the relevant in-domain quality score."

`cases_from_training_jsonl()` — **verified it does not silently skip rows**:
every row that fails is counted in `skipped` and the reason is one of three
explicit branches: JSON decode failure, non-dict row, or
`_extract_qa()` returning `None` (no usable question/answer in any of
ShareGPT `conversations`, OpenAI-style `messages`, or flat `question`/
`answer` keys). `skipped` is returned alongside `cases` and threaded through
into `TrainingEvalMeta.skipped`, which both `build_training_eval()` and
`build_heldout_eval()` include in their returned metadata — so a caller
displaying `case_count` always has `skipped` right next to it, and
`build_training_eval()` raises `ValueError` (naming `skipped`) rather than
returning an empty suite silently if zero cases were extracted.

**Gotcha, not a bug**: `build_heldout_eval()` caps the held-out slice at
`max_cases` (default 200) even if the true 10% split is larger (e.g. a
10,000-row dataset has a 1,000-row held-out slice, but only 200 cases are
returned). This is disclosed — `TrainingEvalMeta.max_cases` is always
present in the returned metadata — but a caller that only reads `case_count`
without also checking whether `case_count == max_cases` could miss that the
held-out score is based on a 200-case sample of a larger slice, not the
whole slice. Worth a route-level UI note if it isn't already surfaced
(outside this lane's file scope — `webui/routes/testing.py` is not in this
lane's file list).

## `audit.py` — independent re-scoring of persisted runs

`recompute_cases(cases)`: takes raw persisted case dicts (e.g. from a DB
row) and re-runs `apply_heuristic_judging()` + `score_results()` from
scratch, **without trusting the stored `verdict` field** — used to catch
drift between what was stored and what the current scoring logic would
produce (e.g. after a scoring bug fix like the ones in this audit, a
re-score surfaces every case whose stored verdict is now provably wrong).
- Malformed stored transcripts (`transcript` field is a string that isn't
  valid JSON) are tracked by case id in `malformed_transcripts`, not
  silently coerced — the case is still re-scored with an empty transcript
  (so it doesn't vanish from the counts) but is flagged for the caller.
- `disagreements`: every case where the stored verdict differs from the
  recomputed one, with both reasonings side by side — this list would have
  been one way to *detect* the `pass_rate` denominator bug above
  post-hoc (any case stored with `verdict=""` whose recomputed score now
  counts toward the total would show up as a disagreement in aggregate
  scores, though `recompute_cases()` itself only diffs per-case verdicts,
  not the aggregate `pass_rate`).
- `passed` field: `True` only if there were zero malformed transcripts *and*
  zero disagreements — a strict "nothing to investigate" signal, not an
  average or threshold.

## `__init__.py`

Empty except for a one-line docstring (`"""Testing subpackage — model
behavior tests."""`). No re-exports; every consumer imports directly from
the submodule it needs (`from finetune_studio.testing.suite import ...`,
etc.).

## Fixes applied in this audit (2026-10-01)

1. **`suite.py`** `score_results()`: `pass_rate` and `weighted_score`
   denominators changed from `n_judged` to `total`, so errored/unjudged
   cases count against the score instead of being silently excluded from it
   (was inflating pass rate — reaches the user-facing `accuracy` field via
   `webui/routes/benchmarks.py:258`, another lane's file, not touched here).
2. **`inference.py`** `_generate_hf()`: now honors the `stop` parameter
   (previously silently ignored for HF/safetensors models while GGUF models
   already respected it via llama.cpp's native `stop=`).
3. **`full_corpus_suite.py`** `cases_from_pairs_directory()`: logs a warning
   per unreadable/malformed pair file instead of silently discarding it.

All three are pinned by regression tests in
`tests/test_testing_engine_module_audit.py` (6 tests, all passing). No
cross-module findings required — all three root causes and their fixes live
entirely within this lane's file list.
