# Finetune Studio — End-to-End UI QA

**File:** `tests/e2e_ui_qa.py`
**Purpose:** exercise every public route in a real Chromium browser, capture
screenshots, assert zero JS errors, and smoke-test non-destructive interactions
against the current Studio UI.

## Run

The suite points at `http://127.0.0.1:7860` by default. Override with the
`FTS_BASE` env var:

```bash
# Live suite against a running WebUI (read-only route and interaction checks)
FTS_ALLOW_LIVE_E2E=1 python tests/e2e_ui_qa.py

# Against an isolated local dev server on a non-production port
FTS_ALLOW_LIVE_E2E=1 FTS_BASE=http://localhost:7891 python tests/e2e_ui_qa.py
```

Outputs:
- Screenshots: `.tmp/qa-shots/ui/<name>.png` (or `FTS_QA_SHOTS`)
- JSON report: `.tmp/qa-shots/ui/report.json`

Exit code is non-zero on any failed check.

The Guide has a fully isolated browser flow; it starts its own sandbox WebUI
and scripted OpenAI-compatible helper, so it never uses the live database or a
real model:

```bash
PYTHONPATH=src python tests/e2e_guide_sandbox.py
```

It checks streaming tool feedback, page navigation, highlighted and pre-filled
controls, conversation persistence, error handling, and both themes.
`e2e_ui_qa.py` checks routes and opens the Files upload panel but does not
upload, train, download, or otherwise mutate project data. Other live E2E
drivers can perform those actions; inspect the selected driver before pointing
it at a non-sandbox service. `FTS_ALLOW_LIVE_E2E=1` remains required for live
runs and for the nightly runner to contact the service or send its report.

## What it checks

For each public route:

1. **load** — HTTP 200 / 3xx, no redirect loop.
2. **no_js_errors** — no `pageerror` events, no `console.error` messages.
3. **design tokens** — the Studio design tokens and system UI font load,
   cards use softened corners, and reduced-motion support is present.

Plus interactive flow tests:

- **dashboard** — polling returns plain text (`Idle` not raw JSON), GPU tile
  shows its ratio, quick actions render, and the brand mark is present.
- **inference** — model `<select>` populated, load button wires up.
- **hf_explore** — search box works, search click returns ≥0 cards.
- **project_data** — bin sprite exists and the upload form opens.
- **spa** — clicking each nav item navigates without a full page reload,
  both `window.spritesInit` and `window.fts.init` are exposed.
- **project sub-pages** — home / files / Smart Brain / testing / training /
  benchmarks / chat / pairs all render cleanly.

## Last recorded run (historical)

`59 / 59 PASS · 0 FAIL` against a live GPU host (2026-09-07).
This is historical evidence, not a claim about the current checkout; rerun the
suite to verify current behavior.

## Testing page (`e2e_track_a.py` / `e2e_user_walkthrough.py`)

The Testing page is a two-step flow and the drivers follow it:

1. **Run** (`#t-run-btn`, mode radios `t-mode` = quiz | rag | dataset) saves only raw transcripts. The drivers wait for
   `GET /api/testing/projects/<pid>/runs/<bid>` to report `status` done, not for a table to appear. An unjudged run has no score:
   `scores.pass_rate` is `null` and every case is "awaiting"; the drivers assert that.
2. **Judge** (`#rv-judge-new` in the run detail) is a separate job. The judge is a model: `FTS_E2E_JUDGE_PROVIDER` picks the provider
   row (empty = the select's default, normally the helper seat) and `judge_status` is polled to `done` (`FTS_JUDGE_TIMEOUT`, default
   1800 s). `auto_judge` is a saved setting that is **off** by default; if it is on, the drivers just wait for the judge it starts.

Scores of record come from the run API and are written to `.tmp/ui-results/<tag>.json` with `judge_model` and `awaiting`; always quote the
judge's name next to a pass rate. Phase `a_review_ui_judge` presses `1` / `3` / `0` on cases and checks that the human verdict
(`judge == "human"`) overrides the AI one and that the judge-vs-human `agreement` follows. Env: `FTS_TEST_MODEL`, `FTS_EVAL_TAG`,
`FTS_EVAL_KIND`, `FTS_QUIZ_TIMEOUT`, `FTS_RUN_BID`. `e2e_ui_qa.py` only loads the Testing page (no run).
