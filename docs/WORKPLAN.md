# WORKPLAN — finetune-studio (durable, model-agnostic)

> Read this FIRST every session, before HANDOFF.md. Written for fresh/dumb models:
> follow steps in order, no reordering, no skipping. If a step is blocked, STOP and
> report the blocker in one line — do not improvise around it.

Last updated: 2026-09-18 (by Amy, ordered by Genor)
Source of truth for ordering: this file. HANDOFF.md describes state, not order.

## Iron rules (every session, every model)

1. **Order is law.** Steps below run strictly in numbered order. Never start step N+1
   while step N is incomplete or blocked.
2. **Fan-dragon runs only the freshest code.** Never start/restart the service while its
   checkout is behind `origin/main`. Sequence is ALWAYS: commit/push here → align
   fan-dragon to `origin/main` → THEN start/restart the service. Stale service up is
   worthless and worse than useless (it hides misalignment).
3. **Never kill foreign GPU/RAM users on fan-dragon** (games, other services). Observe
   only. If headroom is insufficient → pause studio work, one-line report, wait.
4. **Auto-test verdicts are untrustworthy.** Human-grade judging per
   `docs/judging/PROTOCOL.md` is mandatory before believing any suite result. Never
   report auto-scores as truth.
5. **Evidence or it didn't happen.** Report format: `DONE <change> / verified: <pasted
   output> / gaps: <not checked>`. "Should work" is not a status.
6. **Two failures = stop.** Same fix failed twice → state hypothesis in one line, change
   approach or ask one precise question. No circles, no variations of the same edit.
7. **Never restart the OpenClaw gateway. Never SSH-edit files on fan-dragon. Never
   `make run` on genorbox1.**
8. **Rules from this file beat session memory.** This file exists precisely because
   session memory dies between models.

## THE PLAN (in order)

### Step 1 — Finish local WIP, commit, push  [BLOCKED-NO: nothing; current blocker = step 1 itself]
- Repo is dirty: `HANDOFF.md`, `AGENTS.md`, `README.md`, `docs/*`,
  `src/finetune_studio/webui/routes/hf_models.py` (HF search WIP), `tests/test_activity_kind_classifier.py`.
- Finish or consciously commit the HF-models search WIP (it must not be half-broken —
  run `.venv/bin/python -m pytest tests/test_activity_kind_classifier.py -v` and ruff on
  `src/` before pushing).
- `git push`. Only when `origin/main` == local main → Step 2.

### Step 2 — Align fan-dragon to origin/main  [depends: Step 1]
```bash
ssh fan-dragon 'bash -c "cd /home/genortg/finetune-studio && git fetch origin -q && git reset --hard origin/main && git log --oneline -1"'
```
- Record both SHAs (local, fan-dragon). They must match. Only then Step 3.

### Step 3 — Start the service (freshest code only)  [depends: Step 2]
```bash
ssh fan-dragon 'bash -c "systemctl --user start finetune-studio; sleep 5; systemctl --user is-active finetune-studio; ss -ltnp | grep 7860"'
# cgroup truth check (squat uvicorn detection):
ssh fan-dragon 'bash -c "ss -ltnp | grep 7860 | grep -oP \"pid=\\K[0-9]+\" | head -1 | xargs -I{} cat /proc/{}/cgroup | grep finetune-studio"'
curl -sI http://fan-dragon:7860/
```
- If start fails or VRAM/RAM too tight (check `nvidia-smi`, `free -h`, observe only):
  STOP. One-line blocker report. Do not kill foreign pids.

### Step 4 — Smoke the product path  [depends: Step 3]
- `curl` API first, browser only after curl works (browser flakes once → fall back to curl):
  - upload/OCR → parse → use-as-source round trip
  - `/api/activity` shows correct `kind`s after each op
  - HF models search page returns results (verify the WIP from Step 1 live)
- Screenshots: one per WebUI page exercised, taken AFTER the action lands. For Genor's
  eyes, not correctness proof. Correctness proof = pasted curl output.

### Step 5 — Judging session (the goal)  [depends: Step 4]
- Run the held-out suite via API on fan-dragon.
- Pull per-case rows; judge EVERY case per `docs/judging/PROTOCOL.md`:
  expected answer first → model's raw answer → verdict + one-line reason → compare auto.
- Write `docs/judging/2026-09-18-<run-label>.md`: table (case | expected | model said |
  verdict | reason | auto | match?) + summary line `auto says X%, human says Y%, mismatch N/M`.
- Fix nothing in the scorer yet — quantify first.

### Step 6 — Fix worst mismatch classes, re-verify  [depends: Step 5]
- Only after the mismatch table exists. Each fix needs a new judged run showing the class
  is fixed. Then Step 7.

### Step 7 — Later / when headroom allows
- RAG export zip → re-import → re-query round trip (fail loudly if pack useless).
- Training run ≥200 optimizer steps on merged augmented data; held-out only; download
  transcript and READ answers (judging protocol applies).
- Helper GGUF >4B <27B for data-prep/judge seat; unload whatever you loaded before big train.
- Auto-judging automation ONLY after trust gate in `docs/judging/PROTOCOL.md` is met
  (≥95% agreement with human verdicts over 2 runs + Genor spot-check sign-off).

## Blockers log (append, never delete)

- 2026-09-18: fan-dragon service down + behind origin/main. Resolution path = Steps 1→2→3
  in exact order. NOT "just restart the service".
