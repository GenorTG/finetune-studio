# data/prep — QA-pair mining, dataset export, coverage audit

This is the pipeline that turns an uploaded document into training Q&A
pairs and, eventually, an exported JSONL dataset: parse bytes → chunk text →
have the local helper model mine Q&A per chunk → strictly validate →
deterministically backfill any chunk the model missed → dedupe → export.
`data/audit.py` and `data/parsed_edit.py` sit alongside it as the
verification layer and the human-edit hook into the same pipeline.

## `data/prep.py` vs `data/prep/` (the package) — removed dead shim

`src/finetune_studio/data/prep.py` used to sit alongside the
`src/finetune_studio/data/prep/` package and re-export the same public
names (`DataPrepRunner`, `chunk_text`, `export_qa_jsonl`, `parse_qa_json`,
etc.), with a docstring claiming it existed so that old
`from finetune_studio.data.prep import X` imports would "keep working."

**This audit found it was dead code and removed it.** In Python, a package
directory (`data/prep/__init__.py`) always wins over a same-named flat
module file (`data/prep.py`) for the import path `finetune_studio.data.prep`
— confirmed empirically: `import finetune_studio.data.prep as p; p.__file__`
resolved to `.../data/prep/__init__.py`, never the flat file, before the
removal. Every single call site in the repo (`webui/routes/data_prep.py`,
every test file) already did `from finetune_studio.data.prep import X`,
which was always being served by the package — the flat file could never
be reached by any import statement, under any code path, by design of
Python's own import resolution. Its self-referential
`from finetune_studio.data.prep import (...)` line at the top was itself
silently importing from the package, not from itself — a second clue that
this was unreachable rather than load-bearing. Deleted after confirming (a)
the package alone satisfies every real import site and (b) the full
data-prep test battery (82 tests across `test_prep.py`,
`test_data_prep_start.py`, `test_helper_model.py`, `test_versions.py`,
`test_agent_chat_model_resolution.py`, plus this audit's own new tests)
still passes unchanged.

## `data/prep/__init__.py` — public API

Re-exports the stable surface from every submodule listed below (chunker,
export, ingest, parsers, prompts, qa_validate, runner, scorer). This is the
import path the rest of the app (routes, tests) is expected to use:
`from finetune_studio.data.prep import DataPrepRunner`. Its own module
docstring documents the pipeline stage order and file layout — read it first
when orienting in this package.

## `data/prep/chunker.py` — paragraph/sentence-aware text splitter

- `chunk_text(text, target_chars=1200, overlap=200) -> list[str]`: splits on
  paragraph boundaries (`\n\s*\n`) first, falls back to sentence boundaries
  via `_pack_sentences`, then hard-wraps as a last resort. Each returned
  chunk is non-empty — every code path only calls `chunks.append(buf)` when
  `buf` is truthy, so chunk indices downstream can assume a chunk always has
  content *at the time the list is built*.
- `_MIN_STANDALONE_CHARS = 200` — a short buffer (a lone heading) never
  becomes its own chunk ahead of an oversized paragraph; it gets prepended
  instead, because coverage-fill cannot mint a pair from "# Lore" alone and
  that would be a permanent, unfillable coverage hole.
- **Wiring**: called from `ingest.parse_and_chunk` (fresh parses) and
  `audit.audit_source` (to deterministically re-derive expected chunking for
  drift detection). `parsed_edit._rewrite_source_chunks` also calls it
  directly when a human hand-edits parsed text.
- **Gotcha**: the *persisted* chunk list (`files/<sha12>/chunks/NNNN.txt`,
  written by `fs/chunks.py`, outside this lane) is the one `load_existing_chunks`
  (in `ingest.py`) reads back — see the gotcha there about index alignment.

## `data/prep/coverage_fill.py` — deterministic no-chunk-left-behind filler

**The problem it solves** (from its own module docstring): the LLM mining
pass in `runner.py` is stochastic — a chunk whose model output parses to
zero accepted pairs is just skipped, and every progress counter still reads
"done" while that chunk's information never reaches the training set.
`fill_coverage_gaps` is a second, fully deterministic pass: for every
(source, chunk) with no `status="approved"` pair, it extracts verbatim
sentence-level Q&A pairs (question templated, answer quoted exactly) and
re-runs grounding checks before writing them with
`status="approved"`, `origin="coverage_fill"`.

- `fill_coverage_gaps(pid, source_id, sha256="", *, chunk_texts=None, filename="") -> FillResult`
  is the core function. `chunk_texts` maps 1-based chunk index → text; when
  omitted it loads from disk via `ingest.load_existing_chunks(pid, sha256)`.
  `sha256`/`filename` are stamped onto each new pair's `Provenance` (see
  **Fixed** below — they used to be silently dropped even when the caller
  had them).
- `fill_all_project_gaps(pid)` / `fill_sources_gaps(pid, source_ids)` are the
  project-wide and subset-build entry points the export pipeline calls so a
  dataset can never ship with silently-unmined chunks; both aggregate
  `FillResult`s across every source and additionally detect **parsed-artifact
  loss** (`declared_chunks and not chunks` → the source's manifest says N
  chunks but none could be loaded from disk) as its own uncovered-chunk
  reason (`"parsed chunks missing"`).
- `_make_pairs_from_chunk` picks the densest fact-bearing sentences (content
  tokens per char), requires ≥0.9 token overlap between the generated
  answer and the chunk (near-verbatim grounding gate), skips questions
  already seen in this batch, and caps at `_MAX_PER_CHUNK = 3` pairs/chunk.
- `FillResult.chunks_still_uncovered` is the field `DataPrepRunner._run_inner`
  reads (via `fill_summary.get("chunks_still_uncovered", [])`) to decide
  whether to **block export** ("Coverage incomplete… Dataset export is
  blocked."). Anything that doesn't end up in this list is implicitly
  treated as fine by the caller.
- **Wiring**: called from `runner.DataPrepRunner._run_inner` (per-source,
  right after model mining) and from the export routes (project-wide /
  subset, before building the JSONL — outside this lane, in `webui/routes`).

### Fixed (this audit)

1. **Provenance silently dropped.** `fill_coverage_gaps` accepted a
   `sha256` parameter but hardcoded `sha256=""` and `filename=""` when
   building each pair's `Provenance`, even on call paths where the caller
   had both values in scope (`runner.py` has `meta.sha256` and
   `self.filename`; `fill_all_project_gaps`/`fill_sources_gaps` have
   `src.get("sha256")`/`src.get("filename")`). Fixed by adding a `filename`
   keyword parameter and threading both values through at all three call
   sites. No downstream consumer *crashed* on the blank values (one route
   had an existing `row.get("filename") or source.get("filename")`
   fallback), but extractive pairs permanently lacked the same traceability
   model-mined pairs get. Regression test:
   `test_fill_coverage_gaps_stamps_sha256_and_filename_provenance` in
   `tests/test_data_prep_audit.py`.
2. **Documented invariant violated — no-content chunks vanished.** The
   module docstring promises uncoverable chunks are "surfaced as uncovered
   — never silently dropped." But the `not text or not text.strip()` branch
   only incremented `result.skipped_no_content` and `continue`d — it never
   appended to `chunks_still_uncovered`. Since `DataPrepRunner._run_inner`
   only inspects `chunks_still_uncovered` to decide whether to block export,
   a chunk that lost its text entirely (blank chunk, or — see the
   `ingest.py` fix below — a chunk-file gap) silently passed as "covered"
   while never actually getting a pair. Fixed by appending
   `{"chunk_idx": idx, "chars": 0, "reason": "no_content"}` to
   `chunks_still_uncovered` in that branch too (the `skipped_no_content`
   counter is kept for stats). Regression test:
   `test_fill_coverage_gaps_surfaces_no_content_chunk_as_uncovered`.

## `data/prep/export.py` — JSONL exporters

- `deduplicate_qa_pairs(items) -> list[dict]`: returns **one** training
  target per `(source_id, chunk_idx, normalized question)`. Priority order:
  `source-grounded-curated` (3) > default/other (2) > `source-grounded-augmented`
  (1); within the same priority the *shorter* answer wins (a concise target
  beats a whole serialized source record). This was fixed earlier the same
  day to scope the key by `(source_id, chunk_idx)` as well as the question
  text — **verified intact and still correct** by this audit: two different
  sources/chunks can legitimately produce the same auto-generated question
  (e.g. coverage_fill's extractive template reuses a chunk's leading text as
  the question "subject", and two CSVs sharing an identical header row would
  collide on question text alone), and a question-only key would have
  silently dropped the losing source's entire answer. See
  `test_audit_qa_pairs_does_not_flag_legitimate_dedup_as_loss` in this
  audit's test file for a demonstration of the *expected* collapse this
  produces, and why `audit.py` had to learn to account for it (below).
- `export_qa_jsonl(pid, fmt="sharegpt", only="approved")` /
  `export_qa_jsonl_from_sources(pid, source_ids, fmt, only)`: build the
  JSONL for `sharegpt` / `alpaca` / `openai` formats. **Gotcha**: `only="all"`
  does *not* mean literally all pairs — it explicitly excludes
  `status="rejected"` (`items = [q for q in items if q.get("status") != "rejected"]`).
  Rejected pairs are never exportable, even under "all". All three formats
  pipe the answer through `training.clean_answer_for_training` (outside this
  lane) before writing.
- **Wiring**: called from the dataset-export routes (`webui/routes/data_prep.py`
  and friends, outside this lane) after `coverage_fill.fill_all_project_gaps`
  / `fill_sources_gaps` has run to close coverage gaps.

## `data/prep/generator.py` — resolve a chat callable for Q&A mining

Resolves a `ChatFn` (`(messages, max_tokens, temperature, top_p) -> str`)
that calls the **configured local GGUF helper only** — never a different
loaded model (e.g. a project's merged 4B shown on the Inference tab). This
is a deliberate no-silent-fallback contract, documented in the module
docstring.

- `resolve_helper_backend()` tries, in order: (1) `ModelManager`'s active
  provider, if it *is* the helper (`_manager_helper_backend`); (2) the
  global `inference_engine`, if its loaded path *is* the helper GGUF
  (`_inference_helper_backend`). Returns `None` if neither matches — it
  never falls back to whatever happens to be loaded.
- `resolve_generator()` wraps whichever backend resolved into a `ChatFn`
  (`_mgr_chat` calls `mgr.chat(...)`; `_engine_chat` calls `engine.generate(...)`).
- `helper_resolution_error()` distinguishes "wrong model is loaded" (names
  it via `_loaded_non_helper_path`) from "no model loaded at all" for a
  useful UI error message.
- `resolve_loaded_backend(*, prefer_inference=False)` is a **compatibility
  wrapper** that ignores `prefer_inference` entirely and always calls
  `resolve_helper_backend()` — kept only so old call sites that still pass
  that kwarg don't break; it never changes behavior based on it.
- **Wiring**: imported lazily inside `runner.DataPrepRunner._run_inner`
  (`from finetune_studio.data.prep.generator import helper_resolution_error, resolve_generator`)
  right before the per-chunk mining loop starts. Also imports
  `finetune_studio.webui.app` lazily inside `_inference_helper_backend` to
  avoid a module-load cycle with routes.

## `data/prep/ingest.py` — shared parse + chunk steps

Single source of truth for "parse these bytes and write the standard
on-disk artifacts" — used by both `DataPrepRunner` (fresh upload) and the
promote-from-library path (`fs/qa.py`, outside this lane), so both land on
`files/<sha12>/parsed.txt` where `file_library.get_parsed_markdown` and the
RAG build both look.

- `is_already_parsed(pid, sha256) -> bool`: true only when `parsed.txt`
  exists, is ≥10 chars after stripping, and does **not** start with one of
  the parser-error placeholder prefixes (`"[DOC:"`, `"[Failed"`,
  `"[Unsupported"`, `"[EPUB:"`). The length-only check used to treat those
  error strings as "already parsed forever" — fixed 2026-09-18 per the
  comment in the code; this audit re-verified the fix is intact.
- `load_existing_chunks(pid, sha256) -> list[str]`: loads
  `files/<sha>/chunks/NNNN.txt`. **Fixed in this audit** — see below.
- `parse_and_chunk(pid, data, filename, *, sha256, max_chunks=0, mime_type="", reuse_if_parsed=True) -> IngestResult`:
  the actual parse+chunk+persist steps (stages 2–5 of `DataPrepRunner`).
  When `reuse_if_parsed` and the file is already parsed, it skips
  re-parsing and reuses existing chunks (regenerating them from the stored
  text if the chunks directory was itself empty). Otherwise it calls
  `data.parsers.parse_bytes` (the **document-format** parser system — see
  cross-module note below), writes `parsed.txt`/`parsed.json`, updates file
  metadata, chunks the text, and writes the chunks. Returns
  `IngestResult(ok=False, error="empty parse")` if the parsed text is
  <10 chars, or `error="no chunks"` if chunking produced nothing.
  **Gotcha**: `max_chunks` silently truncates the chunk list
  (`chunks = chunks[:max_chunks]`) with no log entry and no warning — but a
  repo-wide grep found **no caller anywhere passes a non-zero `max_chunks`**,
  so this is currently dead config surface, not a live bug. If a caller
  ever does set it, the truncation will be invisible in the ingestion log.
- `ensure_qa_source_parsed(pid, source) -> dict`: parse+chunk a *registered*
  QA source if not already ready; reads bytes from
  `source["data_path"]`/`source["path"]`, content-addresses them via
  `pfs.store_file`, and updates the source manifest to
  `status="ready"`/`status="error"`.
- **Wiring**: `parse_and_chunk` is called from `runner.DataPrepRunner._run_inner`,
  `parsed_edit.reparse_file`, and `audit.audit_source`'s test fixtures.
  `load_existing_chunks` is called from `coverage_fill.fill_coverage_gaps`
  (when `chunk_texts` is omitted) and `fill_all_project_gaps`/`fill_sources_gaps`.

### Fixed (this audit): chunk-index misalignment on a file gap

`load_existing_chunks` used to do `for p in sorted(chunks_dir.glob("*.txt")): out.append(...)`
— it appended in filename-sort order but **never checked that the filename's
own index matched its position in the output list**. Chunks are written as
`0000.txt`, `0001.txt`, … by `fs/chunks.py`. If `0001.txt` were ever missing
(partial write, manual deletion, disk-full mid-loop), `glob()` would return
`[0000.txt, 0002.txt]` and the old code produced `["chunk0 text", "chunk2 text"]`
— a 2-element list. Every caller does `{i: c for i, c in enumerate(chunks, 1)}`,
so chunk **2** would silently be labeled with chunk **3**'s actual text, and
every chunk after the gap would be shifted down by one with no error, no
log, and no way to detect it later (the source's `chunk_count` metadata
would still read 3, undercounting is the only symptom and only if you
compare against that field). Fixed by indexing off each filename's own
`int(p.stem)` and inserting an empty-string placeholder (with a `log.warning`)
for any skipped index, so a gap stays a gap — and now correctly feeds into
the coverage_fill fix above (an empty-text chunk is surfaced as uncovered,
not silently absorbed into a shifted index). Regression test:
`test_load_existing_chunks_preserves_index_across_gap` in
`tests/test_data_prep_audit.py`.

## `data/prep/parsers.py` — parse the **model's own Q&A JSON output**

**Cross-module note (priority check 1, resolved — not a duplicate):**
`data/prep/parsers.py` has nothing to do with `data/parsers.py` /
`data/parsers/` (A1's document-format extraction system for PDF, DOCX,
CSV, etc.). This file parses the *LLM's reply* to the Q&A-mining prompt —
a blob of text that is supposed to be a JSON array like
`[{"q": "...", "a": "..."}]` but in practice arrives wrapped in
`<think>...</think>` blocks, markdown fences, single-quoted "JSON", or as a
numbered Q:/A: list when the model ignores instructions entirely.

- `find_matching_bracket(s, start) -> int | None`: string-aware bracket
  matcher (ignores brackets inside quoted strings, handles escapes) used to
  find where a candidate JSON array actually ends.
- `coerce_pairs(arr, n_expected) -> list[dict]`: normalizes whatever the
  model returned (accepting either `"q"/"a"` or `"question"/"answer"` keys)
  into `{"q":, "a":}` dicts, dropping anything missing either field, and
  truncates to `n_expected`.
- `parse_qa_json(raw, n_expected) -> list[dict]`: the main entry point, four
  fallbacks in order — strip `<think>` blocks (including an unterminated
  trailing one, and prompt-leaked preambles before a bare `</think>`) →
  strip ```` ```json ``` ```` fences → try every `[` position's balanced
  bracket span, parsing each as JSON (and again with `'` replaced by `"` as
  a second attempt) → finally fall back to `parse_qa_lines`.
- `parse_qa_lines(s, n_expected) -> list[dict]`: last-resort regex extraction
  of numbered/bulleted `Q: ... A: ...` lines.
- **Wiring**: `parse_qa_json` is called from `runner.DataPrepRunner._run_inner`
  immediately after the chat call, before `qa_validate.validate_qa_batch`
  runs. Parsing and validation are intentionally separate: this module only
  decides *can we extract a {q,a} shape at all*; `qa_validate.py` decides
  *is it good enough to keep*.

## `data/prep/prompts.py` — system + user prompts

- `QA_SYSTEM_PROMPT` / `QA_USER_TEMPLATE`: the exact strings sent to the
  helper model. The system prompt instructs: JSON array only, no fences, no
  `<think>` blocks, `{"q":, "a":}` shape — directly matching what
  `parsers.py` tries to parse (and has to work around when the model
  ignores the instructions).
- `style_hint(style) -> str`: maps a style name (`socratic`, `factual`,
  `analytical`, `comparative`, `applied`) to a prose instruction fragment;
  unknown styles fall back to `"Balanced, clear Q&A."`.
- **Wiring**: both are formatted per-chunk inside `runner.DataPrepRunner._run_inner`'s
  mining loop (`QA_USER_TEMPLATE.format(chunk=chunk[:6000], n=..., difficulty=..., style_hint=style_hint(self.style))`).
  Note the chunk is truncated to 6000 chars for the *prompt* even though the
  full chunk (potentially up to ~1200 target chars from `chunker.py`, so in
  practice never actually hits this limit) is what gets stored and chunked.

## `data/prep/qa_validate.py` — strict deterministic post-parse gates

Runs **after** `parsers.py` has already produced a `{"q","a"}` shape; this
module decides accept or reject, never touches parsing. Rejection reasons
are stable string keys used as counters: `empty_question`, `empty_answer`,
`malformed_question`, `malformed_answer`, `duplicate_question`,
`unanswerable_from_chunk`, `ungrounded_answer`, `refusal_or_meta`.

- `validate_qa_pair(question, answer, chunk, *, seen_questions=None) -> PairValidation`:
  checks length windows (`8 ≤ |q| ≤ 400`, `8 ≤ |a| ≤ 4000`), minimum content
  tokens, cross-chunk duplicate detection via `seen_questions` (a set the
  caller owns and this function reads but does not mutate — see below),
  **answerability** (question's content tokens must overlap ≥25% with the
  chunk's, or share at least one token when the question has <2 content
  tokens), and **grounding** (answer's content tokens must overlap ≥20% with
  the chunk's, with a refusal/meta-phrase check — `"as an ai"`, `"i cannot"`,
  etc. — short-circuiting straight to `refusal_or_meta`).
- `validate_qa_batch(pairs, chunk, *, seen_questions=None) -> BatchValidationResult`:
  validates a whole chunk's candidate list, **mutating `seen_questions` in
  place** as pairs are accepted (so cross-chunk dedup actually works across
  the whole source — `runner.py` creates one `seen_questions` set before the
  chunk loop and passes the same set into every call).
- `CoverageTracker`: per-source/per-chunk acceptance bookkeeping;
  `uncovered_chunks(chunk_indices)` returns the sorted complement of
  `chunks_with_accepted` — this is what feeds `coverage_fill`'s gap list
  indirectly (via the `covered` set it recomputes from disk, not directly
  from this tracker, since coverage_fill reads `pfs.list_qa_pairs` fresh).
- `build_qa_record(...) -> dict`: assembles the final on-disk pair dict —
  `id`, `source_id`, `sha256`, `chunk_idx`, `chunk_text`, `question`,
  `answer`, `difficulty`, `style`, `score`, `status`, timestamps,
  `provenance` (nested dict, `Provenance.as_dict()`), and a `validation`
  stamp. Note `qa["sha256"]`/`qa["filename"]`... actually only `sha256` is
  mirrored at the top level (`provenance.sha256` → `qa["sha256"]`);
  `filename` only exists inside `qa["provenance"]["filename"]`, not at the
  top level — downstream code reading a pair's filename must go through
  `provenance`, not `pair.get("filename")`.
- **Wiring**: `validate_qa_batch` called from `runner.DataPrepRunner._run_inner`
  right after `parse_qa_json`; `build_qa_record` called there too, and again
  (via a `_SimplePair` adapter) from `coverage_fill._make_pairs_from_chunk`'s
  caller. `content_tokens`/`token_overlap_ratio`/`normalize_question` are
  also imported directly by `coverage_fill.py` and `audit.py` to reuse the
  exact same grounding/dedup definitions rather than reimplementing them.

## `data/prep/queued.py` — lazy per-file prep queue entry

- `QueuedSourcePrep`: holds only a `Path` while queued (no bytes in memory
  for files waiting behind others in a batch upload); `run()` reads the
  bytes lazily and delegates to a real `DataPrepRunner`; `cancel()` either
  cancels the live delegate or, if the job hasn't started yet, flips its own
  lightweight `_queued_progress` to an error state; `progress` proxies to
  the delegate once one exists, else returns the queued placeholder.
- **Wiring**: used by the batch-upload route (outside this lane) to avoid
  holding every queued file's bytes in memory simultaneously.

## `data/prep/runner.py` — `DataPrepRunner`, the orchestrator

The actual pipeline driver for one uploaded file. Stage order: `storing` →
`parsing` → `chunking` → `generating` → `done` (or `error` at any stage).
At each stage it emits a `PrepProgress` callback (SSE UI stream), appends an
entry to `logs/ingestions.jsonl` via `pfs.log_ingestion`, and persists
per-file artifacts.

- `PrepProgress`: a plain dataclass snapshot (`stage`, `pct`, `message`,
  `source_id`, `sha256`, `chunks_total`, `chunks_done`, `qa_total`) pushed to
  `self.cb` on every `_emit` call.
- `DataPrepRunner.__init__`: clamps `qa_per_chunk` to `[1, 10]`.
- `run()` wraps `_run_inner()` in a try/except that logs and emits
  `stage="error"` on **any** uncaught exception — the only place in this
  file that swallows an exception, and it does so loudly (logged + emitted
  to the UI), not silently.
- `_run_inner()`: (1) stores bytes content-addressed via `pfs.store_file`;
  (2) calls `ingest.parse_and_chunk`; on failure, distinguishes
  `"empty parse"` (logs `parse_empty`) from other errors (`"No chunks
  produced."`) and returns `ok=False`; (3) writes the QA source manifest
  with `data_path`/`path` both pointing at the canonical content-addressed
  file — the comment at `runner.py:152-156` documents a 2026-09-18 bug where
  this rewrite used to *drop* `data_path`, causing every re-run after a
  restart to 404 with "source file missing" — fixed and still correct, this
  audit re-verified; (4) resolves the helper via `generator.resolve_generator()`,
  bailing with `helper_resolution_error()` if none; (5) loops chunks,
  calling the model, parsing with `parsers.parse_qa_json`, validating with
  `qa_validate.validate_qa_batch`, writing accepted pairs via
  `pfs.write_qa_pair`, and logging a `qa_chunk_rejected` event whenever any
  pair in that chunk's batch was rejected (with a per-reason breakdown);
  (6) after the loop, calls `coverage_fill.fill_coverage_gaps` to backfill
  anything the model missed, merging its result into the coverage stats
  "honestly" (per the inline comment — it recomputes from the fill result's
  `chunks_still_uncovered`, not a disk re-read, because `pfs` may be
  test-mocked); (7) if any chunk remains uncovered after the fill pass, logs
  a warning, marks the source `status="generated_incomplete"`, and returns
  `ok=False` with an explicit "Dataset export is blocked" message; otherwise
  marks `status="generated"` and returns `ok=True`.
- **Fixed (this audit)**: when `parse_qa_json` returned an empty list (model
  replied, but no `{"q","a"}` pair could be extracted — empty array,
  malformed JSON, pure refusal prose, etc.), the loop did a bare
  `continue` with **no log entry at all** — contrast with the adjacent
  `except Exception` branch three lines above, which *does* log a
  `qa_chunk_error` event for a model-call failure. A chunk that mined zero
  parseable pairs was indistinguishable, after the fact, from a chunk that
  mined some-but-all-rejected (which *does* get a `qa_chunk_rejected` log).
  Fixed by adding a `qa_chunk_unparsed` ingestion-log event (with a 200-char
  raw-reply preview) in that branch, matching the logging discipline used
  everywhere else in this loop. This chunk still correctly gets picked up
  by `coverage_fill` afterward — the fix is about *traceability*
  ("why did this chunk need backfilling?"), not correctness of the final
  dataset.
- **Wiring**: constructed by the upload route and by `QueuedSourcePrep`
  (`queued.py`). Imports `ingest.parse_and_chunk`, `generator.resolve_generator`/
  `helper_resolution_error`, `coverage_fill.fill_coverage_gaps` (lazily,
  inside `_run_inner`, to avoid import cycles), plus `parsers`,
  `prompts`, `qa_validate`, `scorer` at module load time.

## `data/prep/scorer.py` — heuristic quality score

- `heuristic_score(q, a, source) -> float` in `[0, 1]`: starts at 0.5;
  +0.05 for a reasonable question length (8–200 chars); +0.1 for a
  reasonable answer length (30–1500 chars); +0.05 if the question ends in
  `?`; up to +0.25 for lexical overlap between answer and source words;
  −0.3 if the answer contains an AI-decline phrase (`"as an ai"`,
  `"i don't have"`, `"i cannot"` — a small subset of the richer
  `_REFUSAL_PHRASES` list in `qa_validate.py`; this scorer's list is **not**
  the same list and is not kept in sync with it, though both aim at the
  same failure mode).
- **Wiring**: called once per accepted pair in
  `runner.DataPrepRunner._run_inner` (`heuristic_score(accepted.question, accepted.answer, chunk)`),
  stored as `qa["score"]`. Coverage-fill pairs get a hardcoded `score=1.0`
  instead (verbatim extraction needs no heuristic).

## `data/prep/source_state.py` — derived per-file readiness

- `summarize_source(pid, source) -> dict`: computes `stage`
  (`queued`/`parsing`/`error`/`registered`/`training_ready`/`needs_review`/`parsed`),
  `training_ready` (bool), and pair/chunk counts — **entirely derived from
  current `qa_sources`/`qa_pairs` artifacts on disk**, by design (the module
  docstring explains why: a mutable flag would go stale the moment a file is
  reparsed, a pair is rejected, or the source revision changes, so it's
  recomputed every call instead of cached).
- **Gotcha**: `approved_chunks` filters `1 <= chunk_idx <= chunks_total` —
  an approved pair whose `chunk_idx` is out of that range (e.g. stale data
  from before a reparse changed `chunk_count`) is silently excluded from
  the coverage percentage here. This is a *display* computation only
  (powers the UI's progress bar); `audit.audit_qa_pairs` is the place that
  actually flags `chunk_index_out_of_range` as an error.
- **Wiring**: called from the source-list UI route (outside this lane) to
  render per-file pipeline status.

## `data/audit.py` — evidence-grade, deterministic audits

Intentionally never asks a model whether something looks right; it hashes
stored bytes, reparses them, and diffs persisted artifacts against what a
fresh deterministic run would produce.

- `audit_source(pid, source) -> dict`: rehashes the raw file, flags
  `raw_hash_mismatch` if it drifted from the source's recorded sha256;
  reparses via `data.parsers.parse_bytes` (the **document-format** system,
  not `prep/parsers.py`) and flags `persisted_parse_differs_from_deterministic_reparse`
  if the stored `parsed.txt` doesn't byte-match a fresh reparse; rechunks
  via `prep.chunker.chunk_text` and flags
  `persisted_chunks_differ_from_deterministic_chunking` if the stored chunk
  files don't match; also flags `empty_parsed_text`, `no_chunks`, and
  `chunks_have_no_content_tokens`.
- `audit_project_sources(pid) -> dict`: runs `audit_source` over every
  registered QA source; `status` is `"not_applicable"` when there are no
  sources (not `"pass"` — an empty project should never silently read as
  "all good").
- `audit_qa_pairs(pid, *, exported_path=None) -> dict`: checks every
  approved pair's `source_id` resolves to a real source
  (`unknown_source_id`), its `chunk_idx` is in
  `[1, source.chunk_count]` (`chunk_index_out_of_range`), and its answer
  shares at least one content token with its own `chunk_text`
  (`answer_has_no_chunk_token_overlap`). When `exported_path` is given, it
  also validates every line is parseable JSON (`invalid_jsonl_line:<n>`)
  and that the export isn't missing (`export_missing`).
- `audit_suite_cases(suite_path, dataset_path=None) -> dict`: checks a test
  suite file for duplicate case names, and — when `dataset_path` is given —
  cross-checks the suite's case count against the dataset's row count,
  with an explicit metadata-honesty exception for suites that declare
  `coverage == "sampled"`.
- **Wiring**: called from the dataset-export routes (post-export
  verification) and from `tests/test_fidelity_audit.py` /
  `tests/test_suite_full_coverage.py`.

### Fixed (this audit): export-loss check didn't know about legitimate dedup

`audit_qa_pairs`'s export check compared the exported JSONL's line count
directly against `len(pairs)` (every approved pair). But `export.py`'s
`deduplicate_qa_pairs` — by design, and correctly (see the `export.py`
section above) — collapses multiple approved pairs that share the same
`(source_id, chunk_idx, normalized question)` into one training row. Any
time that collapse legitimately happens (e.g. the same question approved
twice across two mining runs), the raw export count would be *lower* than
`len(pairs)` for a completely correct reason — and the audit flagged that
exact, expected shortfall as `"approved_pair_count_differs_from_export_count"`,
indistinguishable from an actual silent-loss bug (a parser crash, a write
failure). That false-positive noise would mask a real regression the next
time one occurred, since "the export count check always fails anyway" is
exactly the kind of alarm people learn to ignore. Fixed by importing
`prep.export.deduplicate_qa_pairs` and computing
`expected_export_count = len(deduplicate_qa_pairs(pairs))` — the same
collapse the real exporter performs — then comparing the actual export
count against *that*, not against the raw pair count. The result dict now
also reports `expected_export_count` and `duplicate_pairs_collapsed` so a
human reading the audit can see *why* the numbers don't match 1:1, instead
of the audit either lying (silently passing) or crying wolf (always
failing). A genuine mismatch (different source/question — see the existing
`test_dataset_audit_flags_unknown_source_and_export_loss` in
`tests/test_fidelity_audit.py`, re-run and still passing after this fix) is
still correctly caught. Regression test:
`test_audit_qa_pairs_does_not_flag_legitimate_dedup_as_loss` in
`tests/test_data_prep_audit.py`.

## `data/parsed_edit.py` — manual parsed-text editing + pipeline status

One concern: a human edits the **parsed** text of a library file (fixing
OCR garbage, trimming boilerplate), and that edit must flow into both
consumers — data-prep/Q&A chunks and the RAG corpus on its next build —
while the original raw bytes stay immutable forever.

- Storage model: the manual override lives at `<raw>.parsed.md` (sibling of
  the stored raw file — note `_override_path`'s own comment: it is
  deliberately **not** `<raw>.md`, because a `.md` upload with the same
  stem would otherwise collide with, and destroy, the original bytes).
  When the file is also a registered data-prep QA source, the canonical
  `files/<sha12>/parsed.txt` is rewritten too and chunks regenerated
  (`_rewrite_source_chunks`), so prep jobs and RAG builds see the edit.
- `_source_for_file(pid, raw, raw_hash="") -> dict | None`: resolves a
  library file to its QA source, preferring a content-sha256 match (the
  source may have been registered from the content-addressed copy rather
  than the raw/ path) and falling back to a path match.
- `save_parsed_override(pid, file_id, text) -> dict`: writes the override,
  invalidates the parsed-markdown cache, and — if a QA source exists for
  this file — calls `_rewrite_source_chunks` (which writes `parsed.txt`,
  rechunks via `prep.chunker.chunk_text`, and best-effort updates file
  metadata, tolerating a missing `metadata.json` for old sources).
- `reparse_file(pid, file_id) -> dict`: discards the override (deletes
  `<raw>.parsed.md` if present) and, if a QA source exists, calls
  `prep.ingest.parse_and_chunk(..., reuse_if_parsed=False)` to force a real
  re-parse from raw bytes — this is the only caller in the whole
  `data/prep` surface that explicitly forces `reuse_if_parsed=False`.
- `corpus_sha12s(pid) -> set[str]`: reads the RAG corpus manifest to find
  which content-sha12 directories are actually indexed. **Gotcha**
  (documented in the function's own docstring): the corpus's internal
  `document_id` is an md5 of the source *path*, not the content sha12, so
  this has to regex-extract the sha12 out of each `documents_meta[].source`
  path string (`.../files/<sha12>/parsed.txt`) instead of using the id
  directly.
- `pipeline_status(pid) -> dict[str, dict]`: per-library-file dict of
  `has_parsed` / `source_id` / `chunk_count` / `parser` / `in_rag`, built by
  cross-referencing `file_library.list_files`/`list_versions` against QA
  sources (by sha256 first, then by path) and `corpus_sha12s`.
- **Wiring**: called from the file-workbench UI routes (outside this lane)
  for the "edit parsed text" and "reparse" actions, and to render the
  per-file pipeline badges (parsed/prep/RAG).

## Full QA-pair lifecycle — count reconciliation, end to end

Traced ingest → chunk → generate → score/validate → coverage_fill → export,
checking that every handoff either preserves the item count or explicitly,
loggedly reduces it:

1. **ingest → chunk**: `parse_and_chunk` chunks the full parsed text; the
   dead `max_chunks` knob aside (never set by any caller today), the chunk
   count written to the source manifest (`chunk_count`) and the actual
   files written under `chunks/` always matched *at write time*. The one
   place this could silently drift after the fact — a chunk file going
   missing later — is exactly the `load_existing_chunks` bug fixed above.
2. **chunk → generate**: every chunk gets exactly one model call; a failed
   call is logged (`qa_chunk_error`) and skipped; a call that parses to zero
   pairs is now logged too (`qa_chunk_unparsed`, fixed above — previously
   silent); a call that parses to pairs runs through `validate_qa_batch`,
   which logs `qa_chunk_rejected` with a reason breakdown for any pair it
   drops. Nothing in this stage disappears without a log entry anymore.
3. **generate → coverage_fill**: `CoverageTracker`/the fill pass's own
   `pfs.list_qa_pairs` re-read determines exactly which chunks got zero
   approved pairs from mining; those are the only chunks `fill_coverage_gaps`
   touches (`gaps = sorted(i for i in chunk_texts if i not in covered)`,
   never re-processing an already-covered chunk — verified by
   `test_fill_only_touches_uncovered_chunks` in `tests/test_coverage_fill.py`).
   The no-content silent-drop bug (fixed above) was exactly at this
   boundary: a gap chunk used to vanish from the uncovered report instead of
   blocking export.
4. **coverage_fill → export**: `deduplicate_qa_pairs` is the *only* place
   the pipeline intentionally reduces the pair count, and it does so for a
   well-defined, documented reason (one training target per
   source+chunk+question). `audit_qa_pairs` now accounts for exactly that
   reduction (fixed above) instead of flagging it as loss.

No other point in this chain was found to drop, overwrite, or truncate data
without a corresponding log entry or explicit "uncovered" record.
