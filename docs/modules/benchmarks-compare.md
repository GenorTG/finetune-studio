# `benchmarks/` + `compare/` — public benchmark suites and model comparison

Purpose: run quantitative evaluation (MMLU / GSM8K / HellaSwag, real HuggingFace splits
and synthetic offline fixtures) against a loaded model, and run the same prompt set
side-by-side across multiple loaded models/APIs for qualitative comparison. This doc
also records a 2026-10-01 dead-code audit: five files in this slice had **zero**
callers anywhere in the app or tests and were deleted.

## Live files

### `benchmarks/real_benchmarks.py` (849 lines) — the real benchmark engine

This is what actually runs when a user clicks "Run Benchmark" in the WebUI for an
official suite, and what `fts benchmark` runs from the CLI.

- `REAL_SPECS: dict[RealFamily, RealBenchmarkSpec]` — static identity (HF dataset id,
  split, official case count, prompt protocol, scoring method) for the three
  supported families: `mmlu` (`cais/mmlu`, split `test`, 14042 official cases),
  `gsm8k` (`openai/gsm8k`, split `test`, 1319 cases), `hellaswag`
  (`Rowan/hellaswag`, split `validation`, 10042 cases).
- `real_suite_path("mmlu")` → `"real://mmlu"` — the virtual path used by WebUI suite
  selection (`real://<family>`); `parse_real_suite_path` / `is_real_suite_path` parse
  it back. `suite_defs.list_builtin_real_suites()` wraps these into `SuiteDefinition`
  rows for `discover_suites()`.
- `format_mmlu_prompt` / `format_hellaswag_prompt` / `format_gsm8k_prompt` — build
  zero-shot prompts. MMLU/HellaSwag ask for a single letter answer; GSM8K asks the
  model to put its final numeric answer after `####` (`extract_gsm8k_gold` extracts
  gold answers the same way from the dataset's own `####`-delimited answer field).
- `mmlu_row_to_case` / `hellaswag_row_to_case` / `gsm8k_row_to_case` — convert one raw
  HF dataset row into a `finetune_studio.testing.suite.BenchmarkCase`. MMLU/HellaSwag
  raise `ValueError` if a row doesn't have exactly ≥4 choices/endings — this is a
  deliberate fail-loud guard, not swallowed.
- `resolve_sample_count` — clamps requested sample count to
  `min(num_samples, MAX_BOUNDED_SAMPLES=500, split_size)`; `full_run=True` or
  `num_samples=None` uses the whole split. Raises `ValueError` if `num_samples < 1`.
- `select_indices` — deterministic sampling: `order="dataset"` takes the first N rows
  in order (used by default); `order="seeded_shuffle"` uses `random.Random(seed)` to
  shuffle `range(split_size)` and take the first N. **Gotcha**: for reproducible runs
  across two calls, the caller must pass the same `seed` — the default `DEFAULT_SEED=0`
  makes this deterministic by default, but a caller that varies `seed` per call will
  get different samples and non-comparable scores.
- `RealBenchmarkSuite` — the main class.
  - `__init__(cache_dir="data/benchmarks/hf_cache", *, dataset_loader=None)` — creates
    the HF cache dir; `dataset_loader` defaults to `_default_load_dataset` (real
    `datasets.load_dataset`) but tests inject a fake loader so CI never hits the
    network — **this is the documented contract**: "Unit tests must inject a fake
    dataset loader — never download HF corpora in CI" (module docstring).
  - `load_cases(family, *, num_samples=50, full_run=False, seed=0, order="dataset")`
    → `(list[BenchmarkCase], BenchmarkMetadata)`. Loads the split, resolves the sample
    count, selects indices, converts each selected row via `_ROW_CONVERTERS[family]`.
  - `evaluate(family, inference_engine, ...)` → `RealBenchmarkResult`. For each case:
    calls `inference_engine.generate(...)`, coerces the response to a string via
    `_coerce_response` (handles both `str` and `{"response": ...}` / `{"text": ...}`
    dict shapes), then scores with `finetune_studio.testing.strict_scoring.score_numeric`
    (GSM8K) or `score_multiple_choice` (MMLU/HellaSwag) — **not** `benchmarks/scoring.py`'s
    heuristic `BenchmarkScorer`; the real/official path uses strict exact-match scoring,
    the heuristic scorer is only for the offline/comparison path (see below).
    `max_tokens` defaults to 256 for GSM8K (needs room to show work) and 16 for
    MMLU/HellaSwag (only a letter is expected).
  - `evaluate_mmlu(..., subjects=[...])` — optional post-load subject filter: loads
    cases, filters by `category in subjects`, **re-derives `meta.sample_count` from the
    filtered list** before re-running via `_evaluate_cases` (the internal path that
    takes prebuilt cases instead of reloading).
  - `run_all(inference_engine, benchmarks=None, ...)` — runs all three (or a requested
    subset) and returns `{"benchmarks": {family: result_dict}, "summary": {...}}`.
    Per-family exceptions are caught, logged via `_log.exception`, and surfaced as
    `{"error": str(exc)}` for that family only — one family failing does not abort the
    others.
- `overall_accuracy_from_run_all(payload)` — the module docstring/comment explicitly
  records a historical bug: "The broken inference endpoint used
  `sum(results.values())` on the nested dict" to try to get an overall score, which is
  wrong because `results` is nested per-family. Callers must use this helper (prefers
  `summary.overall_accuracy`, falls back to manually summing `correct`/`total` across
  families) instead of re-deriving it ad hoc.

### `benchmarks/offline_fixture_data.py` (510 lines) + `offline_suites.py` (133 lines)

Synthetic, hand-authored (not licensed HF data) benchmark cases shaped like
MMLU/GSM8K/HellaSwag, for local comparison without a network/HF-datasets dependency.

- `offline_fixture_data.py` has three builder functions —
  `mmlu_offline_cases()`, `gsm8k_offline_cases()`, `hellaswag_offline_cases()` — each
  building ≥40 cases via small helpers `_mcq` / `_math` / `_completion` that attach
  `keywords` for heuristic scoring (not strict exact-match). Each function asserts
  `len(cases) >= 40` at the bottom as a self-check.
- `offline_suites.py` writes these out as versioned JSON fixture files under
  `benchmarks/fixtures/*.v1.json` via `write_offline_fixtures(force=False)` /
  `ensure_offline_fixtures()` (idempotent — skips writing if the file already exists
  unless `force=True`). `MIN_OFFLINE_CASES = 40` is enforced at write time (raises
  `ValueError` if a builder returns fewer). Every fixture document is explicitly
  labeled `synthetic: True`, `offline: True`, `is_industry_benchmark: False` so nothing
  downstream can mistake it for a real benchmark score.
- Both files are imported by `suite_defs.list_builtin_offline_suites()` (lazy import
  inside the function, not at module top) to discover these fixtures for the WebUI
  suite picker.

### `benchmarks/suite_defs.py` (428 lines) — suite discovery, the WebUI's entry point

- `SuiteDefinition` (frozen dataclass) — one discoverable suite's full metadata
  (`suite_type`: `synthetic_smoke` / `synthetic_offline` / `real` / `local` / `auto`).
  `.label()` builds a human-readable dropdown label per type (e.g.
  `"real · MMLU (official) (HuggingFace · industry) · 14042 official cases"`).
- `list_builtin_smoke_suites()` — tiny (6-case) built-in fixtures, always listed if
  the fixture file exists on disk.
- `list_builtin_offline_suites()` — calls `offline_suites.ensure_offline_fixtures()`
  first (writes fixtures on first use), then lists them.
- `list_builtin_real_suites()` — wraps `real_benchmarks.REAL_SPECS` into
  `SuiteDefinition` rows with `suite_type="real"`, `is_industry_benchmark=True`.
- `discover_suites(project_id=None)` — **the actual function the WebUI benchmark route
  calls** to populate the suite picker. Merges, in order: real HF suites, built-in
  synthetic suites (offline + smoke), known local JSON files under
  `data/benchmarks/*.json` (from a fixed allowlist `_KNOWN_LOCAL_SUITES` plus anything
  else discovered by globbing that directory), and — only when `project_id` is given —
  a project-specific "full ingested corpus" suite and any `auto_suites` DB rows for
  that project. Results are deduplicated into a `dict` keyed by a synthetic key
  (`real:<name>`, `builtin:<name>`, `auto:<name>:<path>`, etc.) and sorted via
  `_sort_key` (real first, then offline, smoke, local, auto).
  - **Gotcha**: for `auto_suites`, rows are queried `ORDER BY created_at DESC` and the
    loop does `if key in found: continue` — the comment explicitly notes this keeps
    the **newest** row per `(suite_name, suite_path)` key, not an arbitrary one.
- `is_selectable_suite(suite_path, project_id=None)` — validates a suite path is one
  of the currently discoverable ones; for non-`real://` paths it also resolves both
  sides to absolute paths before comparing, so a suite selected via a relative path
  still validates correctly against an absolute one returned by discovery.

### `benchmarks/scoring.py` (now ~210 lines, was 230) — heuristic response scoring

`BenchmarkScorer` (and its module-level singleton `scorer`) implements fuzzy,
substring/regex-based scoring: `score_mcq` (6 fallback letter-extraction patterns, from
"letter at string start" down to "choice text appears anywhere in the response"),
`score_math` (extracts a number after `####`/`\boxed{}`/"answer is"/last-number
fallback, then normalizes via `float()` round-trip for comparison), `score_truthful`,
`score_winogrande`, `score_open_ended` (weighted keyword/length/forbidden-word score,
`correct = total >= 0.5`). This is the scorer used by `benchmarks/comparison.py`'s
`_score_response` for open-ended comparison tests — **not** used by the real/official
benchmark path (`real_benchmarks.py` uses `testing.strict_scoring`'s exact-match
scoring instead; see Priority Check #1 below for why these are deliberately different
and both legitimate).

### `benchmarks/comparison.py` (150 lines) — the LIVE model comparison system

`ModelComparator` plus its module-level singleton `comparator = ModelComparator()`.
This is the real side-by-side comparison feature:
- `load_model(name, path)` — loads a local GGUF model via
  `finetune_studio.testing.inference.InferenceEngine` and keeps it in
  `self.engines[name]`.
- `run_comparison(test_suite, config)` — for each test case (accepts either a
  `TestCase` object with `.name`/`.messages` or a plain dict), runs every loaded
  engine on the same prompt, scores via `_score_response` (delegates to
  `scoring.scorer.score_open_ended` when keywords/forbidden words are present, else a
  trivial `{"correct": True, "method": "no_check"}`), and aggregates per-model
  accuracy/avg-time into a `summary` dict.
- **Wiring**: `cli/commands/compare.py` (`fts compare`) and
  `webui/routes/comparison.py` (`POST /compare/load`, `/compare/run`, `/compare/cleanup`)
  both do `from finetune_studio.benchmarks.comparison import comparator` and operate
  on the shared singleton directly — there is exactly one `ModelComparator` instance
  process-wide, so a model loaded via the WebUI is also visible to a concurrent CLI
  call in the same process, and vice versa. There is no `cleanup`/lock discipline
  between the two call sites beyond `engine_guard.ENGINE_LOCK` used elsewhere in
  `webui/routes/comparison.py` for the separate `/rag/chat` endpoint (not for
  `/compare/*`) — concurrent `/compare/load` + CLI `fts compare` against the shared
  singleton is a latent race, flagged below under cross-module findings since the fix
  (locking) would need to touch `webui/routes/comparison.py`, outside this lane.

## `compare/` package — now empty (was a dead duplicate)

`compare/__init__.py` remains (3-line docstring, updated by this audit to explain the
retirement). It imports cleanly and exposes nothing.

## Priority Check #1 — `benchmarks/__init__.py`'s class hierarchy was 100% dead; deleted

The file previously defined `BenchmarkResult`, `BenchmarkSuite`, `BaseBenchmark`, and
13 per-benchmark subclasses (`MMLUSample`, `HellaSwagSample`, `ARCChallengeSample`,
`TriviaQASample`, `WinoGrandeSample`, `IFEvalSample`, `ToolBenchSample`, `GSM8KSample`,
`HumanEvalSample`, `TruthfulQASample`, `PersonaTest`) — 536 lines, each class hardcoding
3–10 toy Q&A samples inline (e.g. `MMLUSample` had a fixed list of 10 trivia questions).

**Evidence it was dead** (full-repo greps, run 2026-10-01, before any edit):

```
$ grep -rn "BenchmarkSuite(" --include="*.py" .
tests/test_real_benchmarks.py:175/214/261: suite = RealBenchmarkSuite(dataset_loader=loader)
src/finetune_studio/webui/routes/benchmarks.py:90: suite = RealBenchmarkSuite()
src/finetune_studio/cli/commands/benchmark.py:22: suite = RealBenchmarkSuite()
src/finetune_studio/webui/routes/chat_v2.py:272: suite = RealBenchmarkSuite()
# Zero matches for the bare `BenchmarkSuite(` from benchmarks/__init__.py — every
# hit above is the unrelated real_benchmarks.RealBenchmarkSuite.

$ grep -rn "BaseBenchmark" --include="*.py" .
# Every hit is inside benchmarks/__init__.py itself (the definition + 12 subclasses).
# Zero external references.

$ grep -rn "from finetune_studio.benchmarks import \|import finetune_studio.benchmarks$" --include="*.py" .
src/finetune_studio/benchmarks/__init__.py:16:  with `from finetune_studio.benchmarks import MMLU` instead of the
# That's the module's own docstring describing an import pattern (`from
# finetune_studio.benchmarks import MMLU`) that was never actually implemented —
# there is no `MMLU` name anywhere in the file. No real caller anywhere imports a
# bare name from `finetune_studio.benchmarks`.

$ grep -rln "finetune_studio\.benchmarks\b" tests/ | xargs grep -n \
  "BenchmarkSuite\|MMLUSample\|HellaSwagSample\|GSM8KSample\|BaseBenchmark\|...PersonaTest"
# tests/test_real_benchmarks.py, tests/test_benchmarks_offline_suites.py,
# tests/test_benchmarks_industry_smoke.py, tests/test_full_corpus_suite.py all import
# only real_benchmarks/offline_suites/suite_defs submodules — zero references to any
# of the __init__.py class names, in production code or tests.
```

The live app's benchmark route (`webui/routes/benchmarks.py`) and the CLI
(`cli/commands/benchmark.py`) both instantiate `real_benchmarks.RealBenchmarkSuite`
directly and use `suite_defs.discover_suites()` for suite selection — never anything
from `benchmarks/__init__.py`. Further evidence this code had never actually been
exercised: `BaseBenchmark.samples: list = field(default_factory=list)` used
`dataclasses.field()` as a **plain class attribute default outside a `@dataclass`**
(the class is not decorated `@dataclass`) — this does not do what it looks like it
does; it leaves `BaseBenchmark.samples` bound to a `dataclasses.Field` object, not a
list. Every subclass happened to override `self.samples` in `__init__`, so this never
surfaced as a visible crash, but it is further evidence this hierarchy was dead
scaffolding, not a maintained execution path.

**Fix applied**: deleted the entire class hierarchy; `benchmarks/__init__.py` is now a
pure docstring pointing at the real submodules (see "Live files" above). No test
referenced any of the deleted names, so no test file needed updating for this part
(the new regression test `tests/test_benchmarks_compare_module_audit.py` pins that
these names are gone).

## Priority Check #2 — `compare/` vs `benchmarks/comparison.py`: confirmed duplicate, deleted

These are **not** two things serving distinct purposes — `compare/engine.py`
(`ComparisonEngine` + `ModelSource` + `ComparisonConfig`), `compare/scorer.py`
(`Scorer` + `ScoreResult`), and `compare/reporter.py` (`generate_report` /
`generate_json_report`) were an entire second, unwired implementation of **the same**
side-by-side model-comparison feature that `benchmarks/comparison.py`'s
`ModelComparator`/`comparator` already provides live. Both systems: load model
sources, run the same prompt across each, score responses by keyword/length/time, and
aggregate per-source summary stats — `compare/engine.py`'s `ComparisonEngine` even
additionally supported remote API sources (`ModelSource.type == "api"` via
`requests.post`) that `benchmarks/comparison.py`'s `ModelComparator` does not, but
nothing ever called it to use that capability either.

**Evidence it was dead**:

```
$ grep -rn "ComparisonEngine(" --include="*.py" .     # zero instantiations anywhere
$ grep -rn "from finetune_studio\.compare\.scorer\|from finetune_studio\.compare\.reporter" .  # zero
$ grep -rn "generate_report(\|generate_json_report(" --include="*.py" . \
  | grep -v "compare/reporter.py\|training/config_optimizer.py\|cli/commands/optimize.py"
# zero — the only other `generate_report` hits are an unrelated same-named method on
# training/config_optimizer.py's optimizer class, not this module.
```

One real, *separate* bug surfaced during this check (reported below, not fixed here —
the fix is outside this lane's files): `webui/routes/quality.py:115` does
`from finetune_studio.compare.engine import FormatConverter` inside the `POST
/convert` route handler — but `FormatConverter` was never defined in `compare/engine.py`
(nor anywhere else in the repo). That import always raised `ModuleNotFoundError`
even before this audit's deletion (the symbol never existed), so every call to that
endpoint has always failed; the handler's broad `except Exception` catches it and
returns `DataJobResponse(status="error", ...)` instead of crashing, so the failure is
silent unless a caller inspects the response body.

**Fix applied**: deleted `compare/engine.py`, `compare/reporter.py`, `compare/scorer.py`
in full (zero callers, zero tests). `compare/__init__.py` docstring rewritten to
explain the retirement and point at `benchmarks.comparison`. `benchmarks/comparison.py`
and `benchmarks/scoring.py` (the live, singular implementation of this feature) are
untouched except for removing one unrelated dead dataclass (next section).

## Additional dead code found and removed (same lane, same pattern)

- **`benchmarks/samplers.py`** (98 lines) — `SamplerConfig` dataclass + `PRESETS` dict
  + `get_sampler()` / `list_presets()`. Zero callers anywhere in the repo, including
  tests (`grep -rn "benchmarks\.samplers\|SamplerConfig(\|get_sampler(\|list_presets("`
  across the whole tree matched nothing outside the file itself; the WebUI's own
  `list_presets` route in `webui/routes/training.py:266` is an unrelated, differently-
  scoped training-config-preset endpoint with the same name by coincidence). Deleted.
- **`benchmarks/tool_calling.py`** (381 lines) — `ToolCallEvaluator`, `AgenticTest`,
  `TOOL_CALL_TESTS`, `get_tool_system_prompt`, `build_tool_prompt_for_model`,
  `detect_tool_format`. Zero callers anywhere in the repo, including tests. Deleted.
- **`benchmarks/scoring.py`**'s unused `BenchmarkResult` dataclass (a second,
  differently-shaped duplicate of the name also defined — and also now deleted — in
  the dead `benchmarks/__init__.py`). Never instantiated anywhere, including within
  `scoring.py` itself. Removed; `BenchmarkScorer` and the `scorer` singleton (both
  live) are untouched.

## Cross-module findings — NOT fixed, needs parent coordination

1. **`webui/routes/quality.py:115-116`** (`POST /convert` handler) imports
   `FormatConverter` from `finetune_studio.compare.engine` — that class never existed
   in `compare/engine.py` (confirmed via full file read before deletion) nor anywhere
   else in the repo (`grep -rn "FormatConverter"` across the whole tree matches only
   those two lines in `quality.py`). This means the `/convert` API endpoint has always
   returned `status="error"` with an import-error message for every call, silently,
   because of the handler's broad `except Exception`. The real conversion logic that
   probably should be wired here lives in `finetune_studio.data.converter`
   (`jsonl_to_json`, `json_to_jsonl`, `csv_to_jsonl`, `simple_to_chat` — module-level
   functions, not a `FormatConverter(source=, target_format=, output=)` class), which
   is what `cli/commands/convert.py` already uses correctly. Fixing this requires
   editing `webui/routes/quality.py` (outside this lane) to call the real
   `data.converter` functions instead of the nonexistent class.
2. **Unsynchronized shared mutable state**: `benchmarks/comparison.py`'s `comparator`
   singleton is used directly, with no locking, by both `cli/commands/compare.py`
   (`fts compare`, a separate process invocation) and `webui/routes/comparison.py`'s
   `/compare/load` / `/compare/run` / `/compare/cleanup` (an always-running WebUI
   process). Within one process, two concurrent WebUI requests to `/compare/run` while
   another request is mid-`/compare/cleanup` can race on `self.engines` (plain dict,
   mutated without a lock) — `webui/routes/comparison.py` already imports
   `engine_guard.ENGINE_LOCK` for its separate `/rag/chat` endpoint but does not use it
   (or any lock) around the `/compare/*` endpoints. A fix would mean adding locking in
   `webui/routes/comparison.py`, outside this lane's file list.
