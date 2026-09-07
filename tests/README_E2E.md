# Finetune Studio — End-to-End UI QA

**File:** `tests/e2e_ui_qa.py`
**Purpose:** exercise every public route in a real Chromium browser, capture
screenshots, assert zero JS errors, validate that the hacker-station aesthetic
stays consistent across the whole app.

## Run

The suite points at `http://fan-dragon:7860` by default. Override with the
`FTS_BASE` env var:

```bash
# Default: against fan-dragon
python tests/e2e_ui_qa.py

# Against local dev server
FTS_BASE=http://localhost:7860 python tests/e2e_ui_qa.py
```

Outputs:
- Screenshots: `~/.openclaw/workspace/media/qa_v2/<name>.png`
- JSON report: `~/.openclaw/workspace/media/qa_v2/report.json`

Exit code is non-zero on any failed check.

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

## Last run

`59 / 59 PASS · 0 FAIL` against `http://fan-dragon:7860` (2026-09-07).
