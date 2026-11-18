---
id: package-usage
title: Package usage, feedback and missing knowledge
page: export
keywords: usage diagnostics errors latency feedback thumbs down missing knowledge gaps privacy retention counts query report
---
## Purpose
See how an installed package is used, and bring candidate knowledge gaps back to Studio as document suggestions.
This is local diagnostics, not token billing, a remote fleet, a signed audit or proof of answer quality.

## Workflow
1. Run a newly exported package in HTTP mode, open its chat page, then Usage & missing knowledge (/usage) on the package host.
2. Enter the package server access token if configured; Load usage creates .usage/owner-token. Read that private file on the
   package host and enter its separate owner token in the page. History/settings/export require X-Usage-Owner and a direct loopback connection,
   a loopback Host, and no forwarding headers. A shared AI client token never grants history access, even if a proxy hides its peer.
   Remote users can give feedback but cannot read history. Neither token is saved by the page or included in reports.
3. Counts only is default. Opting into query storage writes up to 2000 characters in plaintext to .usage/usage.sqlite3,
   outside the encrypted corpus. No answers, passages, credentials, attachments or raw exception messages are recorded.
4. Set retention to 1–365 days (default 30). At most 10000 operations remain; expiry runs on reads/writes, not while stopped.
   Counts cover only retained events. Switching to counts only erases retained question text; Delete retained activity clears events/feedback.
5. Helpful / Not helpful under a package chat response saves one current rating. HTTP query responses carry X-Usage-ID;
   clients may POST /usage/feedback with id and rating up/down. MCP search/answer results expose smart-brain/usage-id in _meta.
6. Export usage JSON or missing knowledge from /usage. Import missing-knowledge.json on Studio's Package page.
   Review the candidate question/reasons, then use Review or add documents. Imported reports are inert and self-reported.
7. Delete imported report removes the Studio copy, including question text. Studio's copy does not expire with package retention.

## Key controls
- `#package-usage` — report intake and diagnostics explanation.
- `#usage-gap-file` — choose a JSON report (at most 32 MiB).
- `#usage-gap-import` — validate and replace this project's last report.
- `#usage-gap-delete` — delete the imported report.
- `#usage-gap-list` — grouped questions, reasons and proposed document-review action.
- `#usage-gap-status` — immediate validation/import/delete feedback.

## What to check
Each outer search/answer/image search is counted once, not again for answer follow-up searches.
Source counts mean distinct retrieved documents per operation, not citations. Low confidence uses the existing retrieval heuristic;
not found also includes an answer reporting that evidence is not stated. Feedback is a signal for review, not a quality verdict.
HTTP validation/provider failures are categorized by status, never raw error text. Retained latency reports mean and p95.
The private sidecar persists restarts and updates; exports contain no activity by default. If storage is unwritable, retrieval
continues but diagnostics can be incomplete. Package/Studio diagnostics stay scoped to their installed package/project.

## Common mistakes
The package passphrase does not encrypt query diagnostics. Keep counts only for sensitive questions.
Exporting or importing a report never fetches URLs, adds files, trains, rebuilds, widens scope or sends telemetry.
Counts-only reports cannot tell you which document is missing: obtain an example question from a user first.
The owner dashboard cannot be administered through a reverse proxy or remote connection; this is not R6 fleet management.
Vendor-account certification, signed identity/audit, provider billing and R7 self-tests are separate work.
