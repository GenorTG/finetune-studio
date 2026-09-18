# Product brief — what Genor wants Finetune Studio to be

Enduring north star for agents (especially fresh MiniMax sessions).  
Ops status and immediate next steps live in root `HANDOFF.md` — rewrite that, don’t bloat this.

## One-sentence goal

Take **original documents → honest training data → a model that answers Q&A about that corpus correctly**, with a WebUI where **every real operation is visible, auditable, and trustworthy**.

## Success looks like

1. **Ingest works** — upload/OCR/parse → “use as source” → data-prep can see and use the file (not “in library but invisible to prep”).
2. **Training actually learns** — enough optimizer steps on grounded data; “completed in 17s” is not success. Source-disjoint / held-out scores matter more than train-set leakage.
3. **Q&A is judged honestly** — automated benchmarks *plus* a human (agent) reading raw transcripts: what the model answered vs expected, not just a green badge.
4. **RAG is first-class** — build corpus, query, **export a package (zip) from WebUI**, re-import, prove the exported pack still works.
5. **Activity feed is complete** — training, prep, RAG, HF download, inference load/chat/unload, benchmarks, export, system update… if it ran, it shows.
6. **UI is clear** — every page/button/result makes sense; broken empty panels, 404 routes, and silent failures are bugs.
7. **Helpers are local GGUF** — data-prep / judge / “helper” inference via GGUF, not a stuck 27B monopolizing VRAM. Prefer something **>4B and <27B** when downloading helpers.

## How to verify (always)

| Layer | Do this |
|-------|---------|
| Local | `make test` / focused pytest + `ruff check src/` on genorbox1 |
| Remote | Deploy → `finetune-studio.service` on fan-dragon :7860 → **cgroup check** |
| API | Prefer `curl` to `http://fan-dragon:7860/api/…` for truth |
| WebUI | Screenshots + click paths; if browser CDP flakes, curl/HTML still counts |
| Quality | Download transcript / audit endpoint; **read answers**; never call leakage “generalization” |
| Activity | Hit `/api/activity` after each op and confirm the right `kind` |

## Held-out / quality rules

- Run **held-out** (or source-disjoint) suites through the WebUI — not train-set echo tests sold as skill.
- Semantic gaps seen before: dispatch IDs, return order IDs, C-17, INC-1842, release facts — fix with **source-grounded augmentation** (`scripts/augment_dataset.py`), not prompt theater.
- RAG can score high while held-out stays weak (~17–26% historically; source-disjoint once 0%) — that is a **training/data** problem, not a “mark it green” problem.
- Optional: secondary model judges each transcript answer after the suite (already partly exposed in benchmarks UI).

## Workflow Genor prefers

1. Fix → test on fan-dragon → evidence (screenshot / curl / transcript snippet).
2. Use WebUI heavily (screenshots), but don’t stall on browser tool flakes — fall back to API.
3. Work **in the parent session** for edits/checks when told; don’t spawn expensive Claude/Cursor fleets unless asked.
4. Clean throwaway OCR/e2e junk projects after probes.
5. **Fan-dragon resources:** high RAM/VRAM is often **other services or games**, not Finetune Studio. **Never** kill/stop foreign processes or interrupt someone else's GPU work. If the box is too tight for train/infer/download: **pause**, one-line blocker, wait for Genor / free headroom. Inside Finetune Studio you **do** manage load/unload (helpers, training target, GGUF prep) so the app does not stack its own models.
6. Empty tool-step chatter. Status ≤8 lines, then **continue the next step in the same turn**.

## Explicit non-goals for a work turn

- OpenClaw tool self-tests after tools are declared fixed.
- Restarting the OpenClaw gateway.
- LLM-compacting a mega-session (start a fresh chat instead).
- Claiming success from suite JSON alone without reading answers.

## Surfaces that must stay healthy

| Surface | Expectation |
|---------|-------------|
| Data-prep + OCR | Real file → parse → source → Q&A / export to training |
| Training | Settings that produce real learning; activity + logs honest |
| Testing | Suites (incl. full-ingested-corpus, RAG-grounded); load model; run; judge |
| Benchmarks | Verdicts on every case; optional secondary judge |
| RAG | Build, query, export zip, re-import, re-query |
| HF models | Search UI shows real results; download GGUF helpers works |
| Activity | Every mutating path classified and listed |
| Theme / SPA nav | Toggle + in-app nav keep JS/buttons alive after click |

## Boxes

| Host | Role |
|------|------|
| **genorbox1** | Edit, pytest, ruff, git. No GPU truth. |
| **fan-dragon** | `:7860` WebUI + GPU. Pull/restart **only the studio service** — never edit app files over SSH. Never kill games/other services for headroom; pause studio work instead. Manage in-app model load/unload. |
