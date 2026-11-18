# Package quality checks

Exported Smart Brains can carry a reviewed QC question set inside their corpus. Encrypted exports
keep questions, answer keys and source IDs inside `corpus.enc`; generic runtime assets expose none.

## Prepare and export

On Studio's **Package** page, paste an existing Testing suite case list (or an object with `cases`)
into **Question set JSON**. Blank ships no tests, including for a new version. There is no hidden
inheritance or automatic question generation. Review the question/answer text for the audience.

```json
[
  {
    "name": "warehouse-price",
    "question": "What is the seasonal warehousing charge?",
    "source_id": "the-exact-document-id",
    "correct_answer": "Nine crowns per season."
  },
  {
    "name": "unknown-policy",
    "question": "What is the lunar freight policy?",
    "expect_abstain": true
  }
]
```

Use `expected_sources: ["id-a", "id-b"]` to require multiple documents. IDs are exact base document IDs,
not substrings or filenames. Every declared source must be in the exported scope; an invalid suite
fails the export rather than silently discarding tests. At most 100 cases, unique names, 4000-character
questions and 8000-character answer keys. No arbitrary code, scripts or user-supplied judge endpoints.

## Run on the installed package

Start HTTP normally and follow **Quality checks** from package chat (`/qc`) on the package host.
Enter its server bearer token if configured and the separate local owner key from `.usage/owner-token`.
The first Load checks request creates that private file if absent. Keys are never persisted by the page.
These operations require direct loopback, loopback Host, same Origin and no forwarding headers.
Sharing a retrieval token does not share access to test keys, transcripts or administration.

**Run retrieval checks** calls the installed search pipeline at its configured top-k. A case is found
only if all expected IDs appear in results. Abstention cases have no retrieval verdict: finding
passages is not proof that those passages answer the question. QC searches do not inflate end-user usage/gap diagnostics. Retrieval never invokes the answer LLM,
but meaning search can use the existing configured embedding endpoint.

**Accept as retrieval baseline** explicitly pins a complete current-suite run with no retrieval
misses. Later runs report found-before/missed-now against it; failures cannot silently replace it.
A changed suite (including answer keys/source IDs) needs a new reviewed baseline. The baseline remains
when the 20-run history rolls over and across process restarts/manual updates.

**Run answers** requires an enabled operator answer model, base URL and model name. Missing setup
returns 503 without an LLM call. Questions and retrieved context go only to that configured provider.
It records raw context/question/answer transcripts; there is no answer verdict yet. **Judge answers**
is a separate request on a saved answer run, with semantic correctness/grounding/abstention instructions,
the answer key and transcript. The judge must return pass/fail/partial and reasoning. Provider errors,
empty answers and malformed judge output are explicit errors and never inferred passes. The same
operator model may answer and judge; this is not independent-model certification.

## Local data and reporting

`.qc/qc.sqlite3` is private **plaintext outside corpus encryption** (0700 directory, 0600 DB).
It retains at most 20 reports plus a pinned baseline, including optional transcripts/context.
It does not store provider configuration/keys or raw exceptions. Model-generated text and reviewed
test content can themselves contain sensitive information: review before exporting **QC JSON**.
**Delete local QC history** erases reports and baseline. No history is included in a new knowledge export.

## Candidate gate contract for future managed updates

`package_qc.update_gate(report, baseline, require_judged=False)` returns `allowed`, `reasons`,
`require_judged`, `scope: candidate_qc_only`, and `fleet_updates: false`.
A missing/incomplete run, missing/incompatible baseline, retrieval misses or regressions blocks. A retrieval-only gate also blocks an abstention-only suite
with no scored source expectations.
With `require_judged=True`, every answer must have a passing judge verdict too.
This is a reusable fail-closed candidate decision interface, **not** R6 fleet management, automatic
update execution, rollback, or enforcement on the existing manual `update.sh` path.

Owner-only HTTP: `GET /qc/report`; `POST /qc/run {"answers": false|true}`;
`POST /qc/judge {"id": "..."}`; `POST /qc/baseline {"id": "..."}`;
`POST /qc/clear {}`. Report export uses the same authenticated owner report.
The runtime is stdlib-only apart from the package's existing retrieval dependencies.
