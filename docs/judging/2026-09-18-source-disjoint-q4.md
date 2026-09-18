# Judged run — source-disjoint held-out suite, 2026-09-18

- **Model:** `output/projects/fbcf7083/runs/quality-v4-source-disjoint/gguf/model-q4_k_m.gguf` (source-disjoint Q4 GGUF export)
- **Suite:** `/home/genortg/.finetune-studio/projects/fbcf7083/source-disjoint-held-out.json` (52 cases)
- **Auto (heuristic) says:** 34.6% pass (18 pass, 2 partial, 32 fail)
- **Amy says (per `docs/judging/PROTOCOL.md`):** **4 pass (7.7%), 8 partial (15.4%), 40 fail** — lenient (partial = ½): **~15.4%**
- **Mismatches auto↔human: 17 / 52 (33%)** — 13 false positives (auto passed wrong answers), 2 false negatives, 2 severity disagreements.
- **Direction: auto OVER-scores this suite.** Wrong dates/IDs/depos passed whenever key words ("2026", "carrier", "unit") overlapped; the bare-answer trap (auto under-scoring correct terse answers) barely fired here.

## Headline: this model does NOT know the corpus. Auto's 35% green is fiction.

Recurring knowledge failures (independent of judging):
- **Person/owner facts systematically wrong** — Mira Varga ≠ "Transport Desk", Pavel Novak ≠ Jonas, wrong risk owners. Auto often passed these on vocabulary.
- **Numbers hallucinated:** cutoff 17:00 (exp. 14:00), revision 2026-09-10 (exp. 2026-08-12), capacity 1,300 (exp. 400-bin), 480 min overtime (exp. >250 units), 2 units (exp. min. 25).
- **Date/ID confusion:** 2026-08-14/18 vs 08-09; CASE-7741 vs CS-491; audit-log actors invented.
- Two robust bright spots: Brno-returns ownership, 2% invoice variance ceiling.

## Case table (verdicts: H=human, A=auto)

| # | Case | Expected (abridged) | Model said | H | A | Why |
|---|------|--------------------|------------|---|---|-----|
| 0 | source-held-000 | same-day, cutoff 14:00 | 120 min after cutoff, 17:00 | fail | fail | both facts wrong |
| 1 | 001 | Rotterdam=Benelux flow, Brno=returns | Brno sorting, Rotterdam distribution, Remote | **fail** | pass | site roles wrong vs ground truth |
| 2 | 002 | Mira Varga, Ops Director | Transport Desk / First Response Owner | fail | fail | wrong person+title |
| 3 | 003 | 06:00–22:00; host + photo ID | weekdays 8–5; signed waiver | fail | fail | hours and doc wrong |
| 4 | 004 | revision 2026-08-12 | revision 2026-09-10 | **fail** | pass | wrong number passed on vocab |
| 5 | 005 | override only after approved cycle-count ticket | "unless discrepancy detected" | **partial** | fail | got half, missed condition |
| 6 | 006 | supervisor approval + reason code | supervisor signs, manager retains copy | **partial** | pass | reason code missing |
| 7 | 007 | supervisor AND finance control | only supervisor | **partial** | pass | half of the answer |
| 8 | 008 | ticket: count sheet, SKU, bin, operator | supplier discrepancy form, lot no., scanner | fail | fail | hallucinated procedure |
| 9 | 009 | >250 units | 480 min overtime | fail | fail | invention |
| 10 | 010 | count sheet, SKU, bin, operator | SKU, scanner, timestamp, operator, reason | **partial** | pass | 2/4 correct + extras |
| 11 | 011 | >25 units | >2 units | fail | fail | wrong number |
| 12 | 012 | 82% of 400-bin cage | 82%, 1,300 units | **fail** | partial | capacity confidently wrong |
| 13 | 013 | 40 bins; recyclables out by Friday | three bins; C-12 debris | fail | fail | wrong facts |
| 14 | 014 | Jonas, 2026-08-14 13:00 | Pavel Novak, 2026-08-18 | **fail** | pass | both facts wrong, auto passed |
| 15 | 015 | threshold 90%, capacity incident | 180 units, warning | fail | fail | wrong |
| 16 | 016 | Lenka, 2026-08-11, lenka@… | dumped employee-table markdown | **fail** | pass | non-answer marked pass |
| 17 | 017 | open capacity incident | check SKU master | **fail** | pass | wrong action |
| 18 | 018 | E118, BX-M, R-14, −3 | Ada Smit, E14A, B-77, 2 | fail | fail | all wrong |
| 19 | 019 | delta −3, bin R-14 | 14 units, B-77 | fail | fail | wrong |
| 20 | 020 | E104, R-801, 2 units | Lenka Horak(=E104), R-801, 42 units | **partial** | fail | person+ID right, count wrong |
| 21 | 021 | 2 units, E104 | two units, Lenka Horak | pass | pass | same person, correct |
| 22 | 022 | conveyor_alert, C-17, system, high | carrier-scan-stale, sftprodc, warning | fail | fail | invented |
| 23 | 023 | E118→E104→system chronology | invented timeline | fail | fail | invented |
| 24 | 024 | CS-491, missing parcel 08-09 | CASE-7741, scan missing | fail | fail | wrong IDs |
| 25 | 025 | arrived Rotterdam, no delivery scan | "shows delivered" | fail | fail | contradiction |
| 26 | 26 | trace opened, no delivered mark | hold on payment, etc. | fail | fail | invented |
| 27 | 027 | Tier One until trace result | Pavel Novak | fail | fail | wrong owner |
| 28 | 028 | 2026-08-09 | 2026-08-14 | fail | fail | wrong date |
| 29 | 029 | Rotterdam depot | Brno | fail | fail | wrong |
| 30 | 030 | matched vs purchase order AND receipt | order lines only | **partial** | pass | receipt missing |
| 31 | 031 | 2% auto-approved | 2% | pass | pass | correct |
| 32 | 032 | routes to procurement | supplier review | fail | fail | wrong |
| 33 | 033 | EUR 500 finance review | 2 percent | fail | fail | wrong |
| 34 | 034 | original invoice number | "an invoice" | pass | pass | semantically equal |
| 35 | 035 | qty→procurement; price below 500, no review | procurement ok; EUR 1,000 threshold claim | **fail** | partial | wrong threshold + wrong outcome |
| 36 | 036 | weekly inspection, Pavel Novak | "C-17 debt", Operations | fail | fail | definition confusion |
| 37 | 037 | RK-03, Vault+MFA | ThermaSafe TS-77 | fail | fail | invented |
| 38 | 038 | 40 bins, Jonas Keller | two bins B-71/72 | fail | fail | wrong |
| 39 | 039 | Elian Mertens, sample inspection | Pavel Novak, inspect before ship | fail | fail | wrong owner |
| 40 | 040 | RK-03 critical, low likelihood | crew injury 7, medium | fail | fail | invented |
| 41 | 041 | RK-01→Pavel, RK-02→Jonas, RK-03→Nadiya, RK-04→Elian | R-77/81/83/92, 3 owners right | **partial** | pass | IDs wrong |
| 42 | 042 | Brno returns | Brno | pass | pass | correct |
| 43 | 043 | named host + photo ID | no credential, ID, sign in | **partial** | pass | host requirement missing |
| 44 | 044 | override after approved ticket | only on scanner-failure cause | **fail** | pass | wrong condition passed |
| 45 | 045 | Mira Varga, Ops Director | "Operations" | fail | fail | auto caught this one |
| 46 | 046 | same-day before 14:00 | 100-min window | fail | fail | wrong |
| 47 | 047 | Rotterdam=Benelux, Brno=returns | Brno both | fail | fail | wrong |
| 48 | 048 | Mira Varga | Transport Desk | fail | fail | wrong |
| 49 | 049 | approved cycle-count ticket | scanner-failure cause | **fail** | pass | wrong condition passed |
| 50 | 050 | Elian Mertens | "Maintenance" | fail | fail | (expected row itself malformed — suite bug) |
| 51 | 051 | >25 units | ≥2 units | fail | fail | wrong number |

## What auto-scoring proves about itself here

- 13 **false positives** — incl. a wrong revision date (004), a garbage table dump (016), and a wrong override condition (049). Root cause: key-word overlap credits tokens like "2026", "risk", "unit" inside factually wrong answers.
- 2 **false negatives** — semantically right answers punished for phrasing (005, 020).
- Net bias on this suite is **+~20 points of phantom accuracy** (34.6 vs ~7.7–15.4).

## Suite-quality bugs found while judging

- `source-held-050`'s **expected answer is a raw markdown table dump** (malformed suite row) — regenerate.
- Overlap between cases 000/046/048, 005/044/049, 006/007/011/051 — same facts asked repeatedly; inflates case count without new signal.

## Recommendation

1. Trust source-disjoint held-out at **~8–15%**, not 35%. Model quality regressed vs the 26.1% disconnected-claim from run `8b1dd006` — verify which export this actually is.
2. Do **not** wire automation yet; see PROTOCOL trust gate. If wiring later, the AI judge must beat key-word overlap on exactly these FP/FN classes.
3. Regenerate the suite: fix malformed expected rows, dedupe overlapping cases.
4. Training fix path stays the same: more optimizer steps + source-grounded augmentation for person/date/number facts (the dominant failure classes).
