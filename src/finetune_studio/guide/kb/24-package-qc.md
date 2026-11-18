---
id: package-qc
title: Exported package quality checks and regressions
page: export
keywords: quality checks QC self tests question suite expected sources baseline regression retrieval judge update gate
---
## Purpose
Ship reviewed questions and expected sources with the knowledge, then test the installed package on demand.
Retrieval checks require no LLM. Answer correctness is decided separately by the operator's AI judge, not keyword matching.

## Workflow
1. On Package, paste a reviewed Testing case list or cases object into Question set JSON. At most 100 cases.
2. Each case needs a unique name, question and exact source_id or expected_sources document IDs.
   Optional answer runs need correct_answer. Unanswerable cases use expect_abstain: true and no expected sources.
   Out-of-scope IDs reject export; also review all question/answer text for the selected audience.
   Blank ships no suite, including on a new version. No implicit suite inheritance is claimed.
3. Build/download/install the encrypted package normally. The suite lives inside corpus.enc, never public runtime metadata.
4. Open Quality checks (/qc) from package chat on the package host. Enter its server token if configured.
   Load checks creates .usage/owner-token if absent; read the private file locally and enter its separate owner token.
   Direct loopback, loopback Host and same-origin/no-forwarding owner rules also apply here.
5. Run retrieval checks. Exact document IDs determine found/missed; abstention cases are not retrieval-scored.
   A source hit never means the answer is semantically correct. Inspect evidence before Accept as retrieval baseline.
6. Run again to compare against the pinned baseline. Found-before/missed-now cases are regressions.
   A changed suite blocks comparability and requires a newly reviewed successful baseline.
7. Optionally configure the operator answer model in rag.config.json or RAG_* runtime configuration.
   Run answers saves transcripts with no verdict. Judge answers separately asks that model for pass/fail/partial with reasoning.
   Missing config is a 503 without an LLM call. Provider/malformed-judge errors remain explicit and unjudged.
8. Export QC JSON for review, or Delete local QC history to erase saved runs and baseline.

## Key controls
- `#package-qc` — package self-test explanation.
- `#mcp-qc-suite` — reviewed question-set JSON sent with this export.
- `#mcp-pkg-btn` — build the package with the selected suite.

## What to check
The owner-only package API is GET /qc/report and POST /qc/run (answers false/true), /qc/judge (id),
/qc/baseline (id), /qc/clear (empty object). It never accepts caller-selected provider URLs or scripts.
The private .qc/qc.sqlite3 sidecar stores plaintext test transcripts outside corpus encryption:
0700 directory, 0600 database, at most 20 runs plus one explicitly pinned baseline. It persists restarts/manual updates.
Suite content remains encrypted; the operator explicitly chooses to send context to the configured LLM for answers/judging.
The candidate-only update_gate contract fails closed on incomplete runs, missing/incompatible baselines,
retrieval misses/regressions and, when require_judged is requested, any answer without a passing AI verdict.
It is an interface for future managed updates, not automatic R6 execution or a manual-update blocker.

## Common mistakes
A retrieved source is not a correct answer. Unanswerable cases need semantic judging, not an empty-hit shortcut.
The same operator model can answer and judge; those are separate requests, not independent model certification.
QC history can contain sensitive question/answer keys and context. Export reports intentionally include them.
Keep owner credentials private, delete local QC history when finished, and do not publish reports casually.
No vendor accounts, remote fleet updates, cloud callback tunnels or new knowledge ingestion are implied.
