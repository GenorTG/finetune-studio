# UI audit plan — 2026-09-16

## Scope

Run the WebUI as a user at desktop and 375px mobile widths. For every route,
capture a screenshot, read the visible content, open each modal, click each
enabled button, and verify the resulting URL, toast, modal, or status change.

## Route matrix

- Global: `/`, `/projects`, `/models`, `/models/explore`, `/inference`, `/settings`.
- Project shell: `/projects/{pid}`, `/data`, `/data-prep`, `/rag`, `/training`,
  `/testing`, `/benchmarks`, `/chat`, `/export`, `/models`, `/settings`.
- Data-prep child surfaces: data editor, source review, QA/review, export and
  ingestion-log links exposed by the page.

## Fixes already applied

- Replace the desktop horizontal nav scrollport with a wrapping tab row.
- Hide the nav strip and expose one hamburger menu on mobile only.
- Remove the visible scroll arrows and scrollbar from the desktop header.
- Bump the stylesheet cache key to `app.css?v=23`.

## Known follow-up checks

1. Verify the new header at desktop, 700px, and 375px; confirm no document or
   header horizontal/vertical overflow and that the hamburger opens every link.
2. Verify `/models` and `/export` table actions at 375px, including copy and
   open-in-inference controls.
3. Verify Training idle/run controls and Chat's empty-state/history controls.
4. Verify Overview and Data Prep file Preview opens an in-page modal rather
   than downloading, and exercise RAW/PARSED/VERSIONS/CONVERSIONS tabs.
5. Verify RAG build/rebuild, source refresh/clear, chunks modal, search, chat,
   bundle download, and delete-corpus confirmation.
6. Verify Benchmark/Testing tabs and disabled-state explanations with no
   dataset/model loaded.
7. Review remaining cramped mobile tables and action clusters; fix only after
   screenshot evidence identifies the offending container.

## Evidence status

The browser-control service was unavailable during this pass because the
gateway was draining, so screenshot and real-click verification remain
blocked. Static template contracts and focused UI tests are green; do not
claim the visual audit complete until the browser sweep is rerun.
