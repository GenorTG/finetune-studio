# R8 — local package diagnostics

TARGET: per-package question/source/gap/error/latency tracking, feedback, privacy and retention,
an owner dashboard/export, and explicit missing-knowledge intake in Studio.

VERTICAL: exported standalone Python runtime and fixed corpus scope already exist; no usage store exists.
Use a stdlib SQLite sidecar in the installed package (never inside the corpus/archive), shared by HTTP/MCP
processes. Count outer user operations, not answer follow-up searches. Store counts by default; query text
requires opt-in. Never store answers, passages, provider configuration, image bytes or exception messages.

HORIZONTAL: package chat and Connect already exist; add owner-only local dashboard, request IDs and thumbs
feedback, JSON export/gap report, and a Studio Package intake which only creates document suggestions.
It must never fetch URLs, add sources, train, rebuild or widen package scope automatically.

BUILD ORDER: store/privacy/retention → runtime instrumentation → owner routes and dashboard → feedback UI
→ Studio validated report intake → Guide/KB/docs → focused regression + actual exported package/browser
→ required CI → squash merge into dev → live health/compute checks.

Scope: local diagnostics and manual report transfer. Remote fleet pairing/telemetry belongs to R6;
vendor-account certification, R7 self-tests and R13 identity/audit are not claimed by this slice.

## Contracts and operating limits

- Exported `package_usage.py` is stdlib-only, explicitly loaded beside `server.py` (also works with Python `-I`).
- `.usage/usage.sqlite3` is local private state, counts-only by default, separate from corpus encryption. Query opt-in
  saves at most 2000 characters. Settings/erasure apply transactionally; 30-day default, 1–365-day range, 10000-event cap.
- Retention prunes on reads/writes (no expiry worker while stopped). Totals cover retained activity, not lifetime usage.
- Outer HTTP query operations and corpus/MCP/CLI search/answer/image calls are measured. Answer-internal searches
  are not extra questions. HTTP validation/provider failures use status categories; raw exceptions never enter storage.
- HTTP events commit before response headers so immediate feedback/report reads see the operation. Disk failures
  preserve retrieval, omit its usage ID and return sanitized 503 for unavailable diagnostics.
- Owner history/settings/exports/deletion need a separate private `.usage/owner-token` via `X-Usage-Owner`,
  bearer auth when configured, direct loopback peer + Host,
  same-origin browser requests and no forwarding headers. Client feedback accepts only opaque ID + up/down.
- JSON exports contain counts and opted-in query text, never source names/passages/answers/credentials/images.
- Studio accepts bounded (32 MiB/10000 events) untrusted reports, stores only grouped inert suggestions atomically,
  with project existence checks and explicit deletion. It never fetches URLs or changes files/index/exported scope.
- Suggestions use the observed question and review reason, not an invented document title or automatic quality verdict.
  Counts-only gaps explicitly ask for a user example. Studio's imported copy persists until deletion/replacement.
- Not included: automatic fleet transfer (R6), provider token/cost accounting, signed identity/access audit (R13),
  vendor certification, automatic document authoring/ingestion, and startup/auth/non-query MCP protocol error audit.

## Verification (2026-11-18)

Final focused package/R8/scope/R2/Guide selection: 122 passed in 69.69 s. Ruff, JS syntax checks,
code-map and diff checks passed. Installer verification: issues none; 140 installed packages compatible.
Actual final encrypted artifact + Studio browser: separate owner key, retrieval/feedback, exact two-operation
counts, query opt-in/retention persistence, real report downloads/upload, counts-only erasure and both deletions.
1270px and 320px inspected screenshots; no horizontal overflow or JavaScript exceptions. Tests use synthetic
data and deterministic providers where needed; they do not certify vendors or production latency.
