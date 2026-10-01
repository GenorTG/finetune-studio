# WebUI frontend — developer reference

This page documents the browser layer under
`src/finetune_studio/webui/{static,templates}`: Jinja-rendered HTML, shared
vanilla JavaScript, and the single stylesheet. API behavior belongs to the
matching `docs/modules/webui-routes-*.md` page. The templates are server
rendered; project pages also opt into link interception via `data-link`.

## Shell and shared assets

| File | Responsibility / wiring |
|---|---|
| `templates/base.html` | Shared document shell: session bar, overflow menu, project flow navigation, resource/activity/status surfaces, theme setup, and cache-versioned shared scripts/styles. Desktop tabs wrap; at ≤700px the compact overflow menu replaces the horizontal strip. |
| `static/css/app.css` | Global tokens and base reset; layout/cards/forms/tables; session bar and mobile menu; command palette/activity/tutorial surfaces; sprite animations; project file browser, wizard and result-table styles; responsive breakpoints and reduced-motion rules. |
| `static/js/app.js` | Shared `window.fts` utilities and initialization for theme, dialogs/notifications, status/resource polling, forms and global controls. Page scripts rely on it; keep initialization idempotent because SPA navigation reruns inline scripts. |
| `static/js/spa.js` | Intercepts same-origin `a[data-link]`, fetches/render-swaps page content, executes page scripts, updates history and scroll. Emits `fts:beforeNavigate` before replacement and `fts:navigated` after initialization; page-owned timers/listeners must tear down on the former. |
| `static/js/nav_overflow.js` | Measures the shared tab groups and maintains overflow affordances / menu contents; loaded with the base shell. |
| `static/js/palette.js` | Ctrl+K command palette, keyboard selection, route/action dispatch. |
| `static/js/activity.js` | Global activity drawer: API/SSE refresh, filters, row expansion, and navigation to related work. |
| `static/js/tutorial.js` | Guided overlay/highlight flow; listens to shared navigation and targets route controls. |
| `static/js/settings.js` | Settings-page interactions and API calls, loaded as a shared page asset. |
| `static/js/training.js` | Training-page shared helpers for run controls/status. Page-specific form wiring remains in the project training template. |
| `static/js/thinking.js` | Shared display helpers for model reasoning/thinking content. |
| `static/js/sprites.js` | Sprite constructors/init hooks used by dashboard, training, ingestion, and benchmark views. |
| `static/favicon.svg` | Static application mark; referenced by the shared shell. |

### Frontend invariants

- `base.html` cache-busts app.css and external JS with `?v=N`; bump the
  matching query version when editing a cacheable asset. Inline template script
  changes are served with the page response.
- SPA replacement must preserve the same script lifecycle as a full load.
  Dispatchers subscribe once or remove subscriptions on navigation; timers
  must be cleared. `_resources.html` now clears its 3-second timer on
  `fts:beforeNavigate`.
- Dynamic strings inserted into HTML use HTML escaping; dynamic values passed
  into inline JavaScript must be encoded as JS literals too. In particular,
  `data_prep.html` uses `jsArg()` for filename/folder names and MIME values,
  and the project-delete confirmation uses `textContent`, not `innerHTML`.
- Keep shared presentation tokens in `app.css` and the light palette block in
  `base.html`. Templates must not invent unresolved CSS variables.
- The stylesheet has intentionally ordered responsive overrides because
  legacy declarations remain above the final navigation/table contracts.
  When a breakpoint change fails to apply, inspect every later rule for the
  selector; equal specificity is resolved by source order.

## Page templates

| Template | Role |
|---|---|
| `index.html` | Global dashboard: host/resource summary, activity and quick entry points. |
| `projects.html` | Project list/search/create/delete; delete confirmation treats stored project names as plain text. |
| `project.html` | Project home, flow picker, dashboard status, and links into model/RAG workflows. |
| `project_wizard.html` | Quick-start orchestration for mining/export/train/test; helper load must target configured helper, and only new pending pairs from successfully mined sources are auto-approved. |
| `project_data.html` | Project file-library browser (upload, folders, tags, versions, trash, use-as-source); distinct from the orphaned legacy `data.html`. |
| `data_prep.html` | Source selection, file actions, pair review/export, parser previews and ingestion state. Dynamic handler arguments are JS-literal encoded. |
| `data_editor.html` | Project dataset row editor with approve/reject/edit actions. |
| `project_training.html` | Project training configuration, dataset picker/upload, launch, and run monitoring. |
| `project_testing.html` | Project test suite and run-result surface. |
| `project_models.html` | Project model artifacts and selection surface. |
| `rag.html` | Portable RAG corpus build, indexed document table, search/chat, shared models, export, clear. Clear removes source text/indexes but retains corpus directory and bundled models. |
| `chat_v2.html` | Project chat, RAG attachments, helper tools, and data-prep chat bridge. |
| `benchmarks.html` | Benchmark suite setup and run/results table. |
| `export_models.html` | Exported model artifacts and export actions. |
| `project_settings.html` | Project-scoped configuration/log view. |
| `hf_models.html` | Hugging Face model search/catalog view. |
| `inference.html` | Global model load controls and inference chat. |
| `models.html` | Local model inventory/actions. |
| `models_index.html` | Global model index and category filters. |
| `testing.html` | Global test-suite runner and result UI. |
| `training.html` | Global training route/template retained for non-project workflows. |
| `export.html` | Global export route/template retained beside project exports. |
| `data.html` | Orphaned legacy template: no `/data` renderer; its multi-file form does not match the singular-file API. Do not describe it as an active page. |
| `settings.html` | Global settings, shortcuts, update/status controls; only list shortcuts implemented by shared JS. |
| `_resources.html` | Reusable host RAM/VRAM component; polling timer is disposed on SPA navigation. |
| `_kv_grid.html` | Reusable key/value settings fragment. |
| `_case_results.html` | Reusable benchmark/test case result table. |

## Verified issues addressed in this continuation

- SPA lifecycle consumers depended on `fts:navigated`, but navigation did not
  emit it; the resource polling interval also survived swaps. Both are fixed.
- The Settings shortcut table advertised Ctrl+S although no save shortcut was
  wired; the false row was removed.
- Project deletion interpolated a database-sourced name into HTML; it now uses
  a data attribute and `textContent`.
- File/folder names in data-prep inline handlers previously relied on HTML
  escaping, which does not protect JavaScript string literals; the calls now
  use `jsArg()`.
- Wizard helper selection and pair auto-approval were too broad; the current
  behavior checks helper identity, run success, and pre-existing QA ids.
- A chat hint named a fixed Qwen helper although the configured helper can
  differ; the copy now says configured helper.
- RAG clear copy and module behavior now agree about source/index removal vs.
  retaining corpus/model files.

## Audit boundary

CSS and the listed browser assets/templates were reviewed during the 2026-10-01
audit continuation, using full template reads from UI lanes and a full
4,995-line stylesheet read. This document is an implementation map, not visual
proof for every route/viewport. The test suite audit is still incomplete; do
not treat static CSS/template assertions as browser-computed-layout coverage.
