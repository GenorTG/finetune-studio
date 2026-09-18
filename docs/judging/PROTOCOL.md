# Transcript judging protocol — the Amy eyeball pass

Why this exists: the auto-scorer (`testing/judge.py` heuristic) is key-word overlap.
A bare correct answer ("1994") against a sentence-shaped expected answer scores ~25% → FAIL
even though it is right. A wrong number inside a chatty answer can score PASS. Auto verdicts
are **evidence for review, never proof**. Until stated otherwise, every held-out / bench
result is only "done" after a human-grade read of the transcript.

## The one rule

Judge like a human holding the answer sheet:

1. **Read the expected answer FIRST.** Know what "correct" means before looking at the
   model's words. Never read the model answer and work backwards to what it "probably meant".
2. **Read the model's raw answer.** Any phrasing counts: bare "1994", "the answer according
   to my memory is 1994", or a three-sentence reasoning chain that lands on 1994 — all correct.
   Speech pattern, length, verbosity, filler: irrelevant.
3. **Verdict on semantics only:**
   - **pass** — the asked-for fact is correct and complete, in any wording.
   - **partial** — right topic, missing part of the asked fact, or vague ("sometime in the 90s").
   - **fail** — wrong fact, wrong number, hallucinated specifics, or no answer.
4. **Record reasoning per case** — one line: what fact was asked, what the model said, why the
   verdict. "pass" with no reason does not count.
5. **Only then compare** against the auto-scorer verdict. Every mismatch goes in the mismatch
   table. The mismatch table is the deliverable that shows how untrustworthy auto-testing is.

## Work rules for a judging session

- **Never judge from suite JSON badges.** Pull the per-case rows (question, expected,
  model_answer, auto verdict) and read them all. Suite-level "87% passed" is not a finding.
- **Screenshot policy:** one screenshot per WebUI page used (suite list, run detail), taken
  *after* the action lands — for Genor's eyes, not as evidence of correctness. Correctness
  evidence is the judged table in `docs/judging/`. If the browser tool flakes once, fall back
  to `curl` immediately (repo gotcha), no CDP retry loops.
- **Reusable commands** (fan-dragon, service must be up):
  ```bash
  # suite runs + per-case rows
  curl -s http://fan-dragon:7860/api/testing/...   # see routes/testing or benchmarks.py for exact path
  # activity check after any op
  curl -s http://fan-dragon:7860/api/activity | python3 -m json.tool | head -40
  # service truth
  ssh fan-dragon 'bash -c "systemctl --user is-active finetune-studio; ss -ltnp | grep 7860"'
  ```
- **Clean up** throwaway probe projects after judging (repo rule).

## Where judged results live

- `docs/judging/<date>-<run-label>.md` — table: case | expected | model said | verdict | reason
  | auto verdict | match?. Plus a summary line: `auto says X%, human says Y%, mismatch N/M`.
- Mismatch table gets copied into the run's HANDOFF notes when it changes the quality picture.

## When automation may replace the eyeball pass (not yet)

Only after ALL of these are true, evidenced over ≥2 full runs:

1. A judge (local helper model via `judge_case_local` / `judge_case_ai`) agrees with Amy's
   verdicts on **≥95% of cases** across two runs, including the bare-answer and chatty-wrong
   cases the heuristic misses.
2. Every heuristic mismatch class found in step 1 has a concrete fix (semantic judge, fact
   extraction) — not threshold tuning.
3. Genor signs off on a spot-check of 10 cases Amy judged.

Until then: heuristic scores are a smoke signal; the eyeball pass is the verdict of record.
