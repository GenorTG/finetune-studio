# Finetune Studio — End-to-End UI QA

**File:** `tests/e2e_ui_qa.py`
**Purpose:** exercise every public route in a real Chromium browser, capture
screenshots, assert zero JS errors, validate that the hacker-station aesthetic
stays consistent across the whole app.

## Run

The suite points at `http://fan-dragon:7860` by default. Override with the
`FTS_BASE` env var:

```bash
# Live suite against fan-dragon (mutates project data and can load a GPU model)
FTS_ALLOW_LIVE_E2E=1 python tests/e2e_ui_qa.py

# Against local dev server
FTS_ALLOW_LIVE_E2E=1 FTS_BASE=http://localhost:7860 python tests/e2e_ui_qa.py
```

Outputs:
- Screenshots: `~/.openclaw/workspace/media/qa_v2/<name>.png`
- JSON report: `~/.openclaw/workspace/media/qa_v2/report.json`

Exit code is non-zero on any failed check.
Live E2E scripts and `run_qa.sh` require `FTS_ALLOW_LIVE_E2E=1`; they can
download models, load GPU memory, train, or mutate project data. The nightly
runner refuses to contact the service or send its Discord report without the
same explicit opt-in.

## What it checks

For each public route:

1. **load** — HTTP 200 / 3xx, no redirect loop.
2. **no_js_errors** — no `pageerror` events, no `console.error` messages.
3. **aesthetic** — computed `border-radius` on `.card` is `0` or `1px`,
   body font doesn't include Inter, hacker-station markers (VT323, JetBrains
   Mono, `prefers-reduced-motion`) all present in CSS, no non-zero
   `border-radius: Npx` slipped back in.

Plus interactive flow tests:

- **dashboard** — boot sprite present, polling returns plain text
  (`Idle` not raw JSON), GPU tile shows ratio, matrix rain renders,
  brand logo has octagon clip-path.
- **inference** — model `<select>` populated, load button wires up.
- **hf_explore** — search box works, search click returns ≥0 cards.
- **project_data** — bin sprite exists, multipart upload returns 200.
- **spa** — clicking each nav item navigates without a full page reload,
  both `window.spritesInit` and `window.fts.init` are exposed.
- **project sub-pages** — home / data / rag / testing / training /
  benchmarks / chat / data-prep all render cleanly.

## Last recorded run (historical)

`59 / 59 PASS · 0 FAIL` against `http://fan-dragon:7860` (2026-09-07).
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
