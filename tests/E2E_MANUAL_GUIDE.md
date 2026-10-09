# Finetune Studio — manual end-to-end test guide (user path)

How to test the app the way a person uses it: real browser, real files, real GPU. Every step says **which page**, **what to
click**, **what to wait for**, **what to check while it runs**, and **what counts as a pass**. The same path is automated in
`tests/e2e_user_walkthrough.py` (phase names in brackets) — run the script to reproduce, read this guide to understand or to
test by hand.

```bash
FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --list
FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --phase create --phase upload --phase prep
```

## Current standard (Genor 2026-10-07) — read this before the numbered steps below

Two independent tracks. **Track A (training)** teaches a model the documents (LoRA adapter / merged model / GGUF). **Track B (RAG)**
indexes the same documents for retrieval and works with *any* model, trained or not. They share nothing but the corpus; never mix
retrieved-context rows into a training dataset for a training test (`grounded_share` off), and never test RAG inside Track A.

- **Corpus:** `tests/corpus/korvane/` — 80 hand-written, messy company documents in 9 lanes (hr, legal, product, it, finance, ops,
  sales, comms, compliance), `core` tier = the 19 files a human reads line by line, `extended` = the rest; `manifest/*.jsonl` is
  the ground truth (4,846 atomic facts). Rebuild with `scripts/corpus_build.py --all`; validate with `scripts/corpus_check.py`.
- **Base model:** **Qwen3.5-9B** (local, loads 4-bit automatically on the 3090). Small models are not valid evidence of quality.
- **Gates are measured, never "at least one":** after upload `scripts/corpus_parse_check.py` (no fact lost by a parser), after
  mining `scripts/corpus_coverage.py` (fact coverage, goal 100 %), after review the same tool with `--status approved`.
- **Review is human, and happens in the Pairs page:** every mined pair is read against its source chunk (`scripts/corpus_review.py dump
  --chunks` prints it) and gets a verdict; the verdicts are then **replayed through the Pairs page with the keyboard**
  (`apply --ui` records them, `--phase a_review_ui` replays A / R+reason / edit / add). A pair without a verdict stops the run: no
  "approve all". `plan --ui` pre-fills the verdicts of pairs identical (file, chunk, question, answer) to the golden review in
  `tests/corpus/korvane/golden/`; everything else is NOVEL and must be read. Facts no approved answer states yet are added in the UI
  (`--phase a_adds_ui`, reads `.tmp/missing_adds.json`).
- **Scoring happens in the Testing page**, in two separate steps: a test RUN only saves raw transcripts (question, the model's answer,
  the answer key) and has **no verdict and no score**; correctness is decided afterwards by a **judge**, an AI judge (any provider row,
  default = the helper seat) or you. Nothing is scored by string matching. `--phase a_test_ui` imports
  `tests/corpus/korvane/eval/korvane_quiz_core.jsonl` (102 questions + 20 `expect_abstain` ones that should be declined), picks a model,
  presses **Run test**, waits for the run to finish through the run API, then presses **Judge unjudged** in the run detail (judge:
  `FTS_E2E_JUDGE_PROVIDER`, empty = the select's default) and reads the scores off the run API. `--phase a_eval_ui` does the same in
  **Dataset check** mode (`FTS_EVAL_KIND=training_leakage` = memorization control, `heldout`); `--phase b_rag_quiz_ui` in **RAG-grounded**
  mode; `--phase a_review_ui_judge` presses keys `1` / `3` on two cases and checks the human verdict and the agreement in the API.
  Use the **q4_k_m GGUF**: the merged bf16 through the Testing page took > 40 min for 122 questions. `scripts/corpus_eval*.py` remain as
  cross-checks only.
- **Reporting a score:** an unjudged run is "awaiting" (no pass rate), `auto_judge` is a saved setting and is **off** by default, and the
  judge is a model, so **always name it next to the number** ("62/122 = 50.8 % judged by `<judge_model>`"; the phases write `judge_model`
  and `awaiting` into `.tmp/ui-results/<tag>.json`). A score from a different judge is a different measurement; where you disagree you
  override the verdict yourself (below) and the run shows how often the judge agrees with you.
- **Training gate (revised after the 2026-10-08 run):** NOT "final loss < 0.5" and NOT eval-loss minimum. Unsloth: training loss ≈ 0.5–1.0 is
  healthy and < 0.2 suggests over-fitting, but for facts the eval loss of a random pair hold-out *rises* while paraphrase recall keeps
  improving (early stopping kept the epoch-2 weights: 12 % vs 18 % recall at 6 epochs). The gates are **paraphrase recall**
  (`scripts/corpus_eval.py`: 102 re-worded questions about trained facts) and **abstention** on 20 unanswerable questions, measured on the
  q4_k_m the user will run (and on the bf16 merge when judging training quality). Baseline numbers: `tests/corpus/korvane/RESULTS.md`.
- **Reviewed preference pairs:** the in-app abstain builder produces mostly answerable questions (112/150); check every one against the whole corpus.
- One GPU job at a time. Delete projects, exports, checkpoints and merged models when the run is done.

```bash
FTS_ALLOW_LIVE_E2E=1 FTS_PROJECT_NAME=korvane-ui-1 .venv/bin/python tests/e2e_track_a.py --phase create --phase a_upload --phase a_prep   # mining ~1 h
# read: scripts/corpus_review.py plan --ui ; dump --chunks ; apply --ui / add --ui   (verdicts -> .tmp/ui-verdicts.jsonl)
FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_track_a.py --phase a_review_ui          # replays every verdict in the Pairs page (~5 min for 3.7k)
.venv/bin/python scripts/corpus_coverage.py --pid <p> --status approved                  # missing facts -> .tmp/missing_adds.json
FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_track_a.py --phase a_adds_ui            # until coverage is 1130/1130
FTS_ALLOW_LIVE_E2E=1 FTS_EPOCHS=6 FTS_EARLY_STOP=0 .venv/bin/python tests/e2e_track_a.py --phase a_export --phase a_train   # ~45 min
FTS_ALLOW_LIVE_E2E=1 FTS_EXPORT_TIMEOUT=2400 .venv/bin/python tests/e2e_user_walkthrough.py --phase gguf                    # q4_k_m + q6_k, ~8 min
FTS_ALLOW_LIVE_E2E=1 FTS_TEST_MODEL=q4_k_m FTS_EVAL_TAG=sft6-q4km .venv/bin/python tests/e2e_track_a.py --phase a_test_ui     # ~12 min
FTS_ALLOW_LIVE_E2E=1 FTS_TEST_MODEL=q4_k_m FTS_EVAL_KIND=training_leakage FTS_EVAL_MAX=200 .venv/bin/python tests/e2e_track_a.py --phase a_eval_ui
FTS_ALLOW_LIVE_E2E=1 FTS_RUN_ID=<run> .venv/bin/python tests/e2e_track_a.py --phase a_chat --phase b_rag                   # Chat load + Track B index/search
FTS_ALLOW_LIVE_E2E=1 FTS_TEST_MODEL=q4_k_m FTS_EVAL_TAG=sft6-q4km-rag .venv/bin/python tests/e2e_track_a.py --phase b_rag_quiz_ui
FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_track_a.py --phase a_guide --phase a_cleanup
FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_track_a.py --phase a_dpo_build --phase a_dpo_train   # not re-run in the UI-only pass; review the pairs first
.venv/bin/python scripts/bench_review_page.py                                             # review-page latency on the golden pairs (scratch project)
```

(The numbered steps below are the older generic walkthrough, still valid for the page-by-page details; `rag` there is Track B.)

It creates a throwaway project (`ux-walk-1`) on the **live** service and the `cleanup` phase deletes it with its models. Never
use a project you care about. Screenshots: `.tmp/qa-shots/manual-e2e/`.

## 0. Before you start (5 min)

| Check | How | Expect |
|---|---|---|
| Service up, build is current | `systemctl --user is-active finetune-studio`; `curl localhost:7860/api/system/version` | `active`; version = `VERSION` file |
| GPU is the 3090 only | `curl localhost:7860/api/system/accelerator` | `kind: cuda`, `RTX 3090`, `device_count: 1`, no `degraded_reason` |
| Parsers complete | `bash install.sh --check` | no `document parsers missing` issue |
| VRAM is free | `nvidia-smi -i 0` | < 2 GiB used (foreign processes may hold ~1 GiB) |
| Helper model present | `curl localhost:7860/api/providers/helper/status` | helper GGUF exists |
| Logs | `journalctl --user -u finetune-studio -f` in a second terminal | keep it open the whole time |

**Keep three things visible while any long step runs** (the script's monitor prints them):
1. `nvidia-smi -l 3 -i 0` — VRAM and utilisation (helper load ≈ +12 GiB, training ≈ +4–6 GiB, util > 50 %).
2. `journalctl --user -u finetune-studio -f` — no `Traceback`, `CUDA error`, `OutOfMemory`; a `Main process exited` line is a crash.
3. The header chip (top right): `no model` → model name while a model is resident. It must never lie.

**The app must stay responsive during long jobs.** While the helper loads (≈ 50 s) or trains, click around: open another page,
reload. If a page takes longer than ~2 s to answer, that is a bug (it was one: helper load froze the app for ~55 s).

## 1. Create the project [`create`]

Page `/projects` → **＋ NEW PROJECT** → name `ux-walk-1`, a description → **CREATE**.
- Wait: card appears (page stays on `/projects`). Click the card → project overview.
- Check: the card shows the id; overview links (files, pairs, training…) all work.
- Pass: exactly one project via `GET /api/projects`.

## 2. Upload mixed documents [`upload`]

Page `/projects/<id>/data` → **⬆ UPLOAD** → **Choose Files** (select all 14 files of
`.tmp/manual-e2e/corpus/`: txt, md, docx, **doc**, pdf, html, csv, xlsx, **xls**, pptx, rtf) → **click UPLOAD** (choosing files
does not start the upload).
- Wait: ≤ 10 s; every row gets a green **PARSED** pill.
- Check each row: TYPE shows `DOCX`/`XLSX`… (not a long MIME string); size plausible; open the eye icon → parsed text is
  complete and readable (tables flattened, no HTML tags, no duplicated title).
- Pass: every file `status: ready`, `char_count > 0`, parser name sensible (`pdf_v1_pypdf`, `docx_v1`, `html_v1`…).
- Failure signs: empty text for `.xls/.pptx/.rtf` (missing parser module), a row stuck on `queued`.

## 3. Mine Q&A pairs with the agentic helper [`prep`]

Page `/projects/<id>/data-prep` ("pairs"). In **Parsed sources** tick every file → **GENERATE PAIRS FOR SELECTED**.
**Wait (about 1.5–2 min for 14 small files)** and watch, in this order:
1. `nvidia-smi`: VRAM jumps ≈ +12–14 GiB within ~50 s (helper `gemma-4-12b … Q4_K_M` loading), header chip shows the model
   `gpu 48/48`.
2. Utilisation 60–90 % while it writes pairs; the table rows change `PARSED → TRAINING READY` / `NEEDS REVIEW` (press **REFRESH**
   — the table is not live).
3. Journal: no errors; each source logs a `qa_generated` event (`GET /api/projects/<id>/data-prep/ingestion-log`).
- After helper auto-unload (wait for `GET /api/inference/status` → `loaded: false`), start another selected-source job. Pass only if
  the helper reloads before mining (`active.loaded: true`, VRAM rises); a job that says `done` while the journal says
  `Local model not loaded` is a failure even if coverage-fill rows were produced.
- Pass: every source has ≥ 1 pair; helper pairs are **pending** (need review), coverage-fill pairs **approved**.
- Quality checks (do these by reading, not by trusting counts): questions are self-contained and specific; answers are
  verbatim/faithful to the source (spot-check 10: a number or name that is not in the file is a hallucination); no pair opens
  with a pronoun or "the above"; no chunk starts mid-word; vague files (`scan_0042.txt`) yield vague or no pairs — that is the
  validator working, not a bug. The log's `rejection_counters.by_reason` (`unanswerable_from_chunk`, `ungrounded_answer`) shows
  what the grounding filter dropped.
- Industry check: pairs are generated *per chunk* with a difficulty/style (`socratic`, `factual`, …). For facts that must be
  memorised, generate ≥ 3 pairs per fact in different phrasings (paraphrase diversity is what lets a model *extract* a fact
  later — Allen-Zhu & Li, "Physics of Language Models 3.1").

## 4. Review like a curator [`review`]

Same page, **Training / Q&A output** → filter **PENDING**.
- Reject what a human would reject: tick the vague/ambiguous rows → **REJECT SELECTED**. Edit a pair by clicking it if the
  question is fine but the answer is sloppy.
- Then **APPROVE ALL PENDING**. Counters (`TOTAL / PENDING / APPROVED / REJECTED`) must add up.
- Pass: 0 pending, rejected pairs excluded from the export.

## 5. Make it searchable — RAG [`rag`]

Page `/projects/<id>/rag` → **⚡ QUICK INDEX**. Wait ≈ 15 s (first run downloads the embedder).
- Check: documents/chunks counters; **SEARCH** "How long does the Aurora Kettle warranty last?" → top hit is the warranty
  paragraph and shows its **source filename** and score. Ask 6 fact questions, one per document type; all must hit.
- Do this **before** exporting, because the next step can add retrieved context to training rows.

## 6. Build the training dataset [`export`]

Back on `/data-prep`: leave **Include retrieved context** ticked (≈ 40 % of rows get the RAG-chat prompt + their own source
passage) → **EXPORT APPROVED → TRAINING**.
- Wait: seconds. A green line "Exported N approved pair(s) (M with retrieved context)" and a **START TRAINING WITH THIS
  DATASET** button appear.
- Pass: dataset registered (`GET /api/projects/<id>/datasets`) with `qa_count` = approved pairs. If a file produced no usable
  pair the export is **blocked** with a 409 naming the file: delete the file or export without it (the Quick-work wizard has
  buttons for both).
- CLI equivalent: `fts dataset build --project ux-walk-1 [--force] [--json]`.

## 7. Get a base model [`model`]

Page `/models/explore` → search `Qwen3-0.6B` → on the `Qwen/Qwen3-0.6B` card **DOWNLOAD** → confirm **OK**.
- Wait ≈ 20 s (1.5 GB). Pass: `GET /api/hf/local` lists it; it appears in the Training base-model dropdown.

## 8. Train [`train`]

Page `/projects/<id>/training`: **Base model** = Qwen3-0.6B; **From this project** → the exported dataset; **Preset** =
*Precision — very good for factual/domain recall* (fills rank 64, lr 2e-4, 60 epochs, batch 2, seq 2048); **START TRAINING**.
- Wait ≈ 4 min for ~420 steps on the 3090. Watch: header chip switches to training, **the helper is unloaded automatically**
  (VRAM drops from ~18 GiB to ~1 GiB, then rises to 5–6 GiB), live panel shows step/loss/ETA, loss falls from ≈ 3 to < 0.1.
- Pass: status `done`, final loss < 0.5, a merged model exists, the run is listed with its output path. Early stopping is
  ticked by default: with a tiny dataset make sure it did not stop before the planned steps unless validation loss really rose.
- Failure signs: OOM → the page must show "Out of GPU memory" with the knobs to change (batch, max sequence), VRAM must return to
  baseline and the service must stay up.

### Preference-tuning branch (DPO)

Use this branch only when you have **comparisons for the same prompt** and a reviewer can explain why one answer is preferred.
Preference data is not interchangeable with Q&A: a chosen answer by itself is SFT, while a pair of chosen/rejected answers is DPO.

0. **Author the pairs in-app (optional).** After SFT, on `/projects/<id>/data-prep` open the **Preference pairs (DPO)** card, load the
   helper, keep both kinds ticked, click **Build preference pairs**. Watch the progress bar (every candidate is a helper call), then read
   the summary: pairs per kind, dropped-candidate reasons, chosen/rejected length ratio (a ⚠ means chosen is systematically longer —
   DPO can learn length instead of faithfulness). Spot-check ~10 pairs in the JSONL (`<datasets dir>/<pid>-preference.jsonl`): hallucination
   `rejected` should invent or omit details the source states; abstain questions must NOT be answerable from the files. **Train with this
   dataset →** opens Training with the DPO route and the dataset preselected. CLI twin: `fts dataset build-preference --project <id> --json`.
1. Otherwise prepare JSONL with at least two rows. Conversational form is preferred:
   `{"prompt":[{"role":"user","content":"…"}],"chosen":[{"role":"assistant","content":"…"}],"rejected":[{"role":"assistant","content":"…"}]}`.
   String fields (`prompt`, `chosen`, `rejected`) are accepted and normalized to user/assistant turns.
2. On `/projects/<id>/training`, select **Preference tuning (DPO)**, then **Upload my own** and choose that JSONL.
3. Check the example/intent before starting: each row must have a non-empty prompt, different non-empty chosen/rejected outputs,
   and the prompt ends with a user turn. Invalid data must return a visible 400 without loading a model or starting the worker.
4. Choose the same Transformers-compatible base-model type used by SFT and start. The run detail must record `training_mode=dpo`;
   progress, Stop, adapter save, merge and optional GGUF export should use the standard run lifecycle.
5. When the run finishes, its auto-generated quiz asks each prompt and expects **chosen**, never rejected. Inspect these answers;
   then run held-out evaluation. Do not infer model quality from training loss alone.

The DPO radio selects measured LoRA defaults (learning rate `1e-4`, 3 epochs, no warmup, plus a keep-chosen NLL term; TRL's `1e-6` moves nothing on LoRA); these values appear in the editable fields. Review comparisons before training. The first preference route is DPO. ORPO/KTO and a per-pair review UI for preference comparisons are not included yet (pairs are authored in bulk with quality gates, not hand-reviewed).

### Other supported data routes

- **Tool-calling SFT:** upload JSONL whose rows contain a `messages` array (assistant `tool_calls`, corresponding `tool` result turns, and a final assistant turn) plus a `tools` array of JSON function schemas. Select **Tool-calling SFT**. The chosen tokenizer must have a chat template that accepts those schemas. Verify rendered examples before a large run; this trains tool-call syntax, not the external tool runtime or its permissions.
- **Continued pretraining:** upload JSONL with one `{"text":"…"}` raw-text row per sample and select **Continued pretraining**. Do not wrap corpus text as a fake user message. Keep a held-out domain-text set and downstream task suite; this route is not instruction tuning and has no automatic QA quiz.
- **Reasoning distillation:** upload reviewed teacher demonstrations as `messages` JSONL and select **Reasoning distillation**. Keep a final answer / verifiable outcome in the example, remove unsupported or private teacher traces, and evaluate on independent tasks. It uses the normal supervised trainer; the label does not make traces trustworthy.

ORPO/KTO are not exposed: the installed TRL 1.14.1 runtime provides DPOTrainer but no ORPOTrainer/ORPOConfig. Do not label DPO as ORPO or silently substitute algorithms. Automatic teacher-trace generation, and benchmark decontamination are also not in-app yet.

## 9. Test the trained model [`test`]

Page `/projects/<id>/testing`. Testing is **two steps**, and only the second one produces a score.

**1 · Run.** Pick the mode (**Project quiz**, **RAG-grounded**, **Dataset check**), the **Model under test** (default = latest merged
export) and the quiz (the auto-generated `<project>-sharegpt-approved` suite, or import your own JSON/JSONL under *Bring your own quiz*),
then **Run test** (≈ 40–90 s for 61 cases; the model is auto-loaded first). The run opens below as `?run=<id>`; it saves each raw answer
as it goes. A finished run shows **awaiting** everywhere: no verdicts, no pass rate. Nothing is compared by string matching.

**2 · Judge.** In the run detail pick the judge model (**Judge this run**; default = the helper seat, any connected provider row works)
and press **Judge unjudged** (**Re-judge all** asks again; earlier AI opinions stay for comparison). The summary chips then show
`judged / awaiting / pass / partial / fail / pass rate`, with the judge named under *Last judge*. The checkbox *judge automatically after
each run* (`auto_judge`, saved, **off** by default) makes step 2 start by itself after step 1.
- **Override a verdict yourself:** open a case (list on the left, pane on the right) and press `1` pass, `2` partial, `3` fail, `0` clear
  your verdict (or use the buttons; add a note in the field). Your verdict always wins over the AI's; the chips then show
  *reviewed by you* and how often each judge agrees with you. Filters: all / awaiting / pass / partial / fail / not reviewed by me / mine.
- Read it correctly: plain rows are *recall of training facts*; grounded rows are asked with their own context (chips **from memory** vs
  **answering from retrieved context**). Neither is generalisation. The honest generalisation number is a **Dataset check → Held-out
  slice** run (the seed-42 10 % slice the trainer never saw).
- Check a few cases by hand even when an AI judged them: answer key vs model answer vs the judge's reasoning, and watch for
  training-data bugs showing up as the key (e.g. an answer cut at `Dr.`). **A pass rate is only as good as its judge: state the judge's
  name with every number you report.**
- API (what the E2E phases poll): `GET /api/testing/projects/<pid>/runs/<bid>` → `status` (running|done|failed|cancelled),
  `progress_done/total`, `judge_status` (''|running|done|failed|cancelled), `judge_model`, `scores{total,judged,awaiting,passed,partial,
  failed,pass_rate,retrieval}`, `agreement`; `.../cases` lists cases with `verdict`, `judge` (`ai`|`human`|`none`) and `judgements[]`.
- Also try **RAG-grounded** mode (retrieval then answer; the run also records retrieval metrics) and **Dataset check → Full training set**.

## 10. Benchmark [`bench`]

Page `/projects/<id>/benchmarks`. In the **trained run row** pick *synthetic · knowledge MCQ (offline)* → **RUN** (≈ 20 s).
The other row is the **base model** — its official-suite runs (GSM8K/MMLU/HellaSwag, sampled 50) download datasets and are slow;
do those deliberately, not by accident. Scores appear under *Recent scores*; *Compare two runs* shows tuned vs base.
Offline suites are smoke checks, **not** comparable with published numbers (the page says so).

## 11. Export a GGUF [`gguf`]

Page `/projects/<id>/export`: pick the finished run; GGUF + `q4_k_m` are pre-ticked (recommended); also tick `q6_k` → **EXPORT
SELECTED**. Wait ≈ 20–40 s. Pass: two files under `output/projects/<id>/runs/<run>/gguf/`, listed under *Trained exports*, selectable
in Chat and Testing. Prefer q6_k/q4_k_m for small models — see AGENTS.md Gotchas for the Q8_0 history (the loader now caps the
micro-batch, but there is no reason to pick the riskiest quant for a 0.6B model).

## 12. Chat and agent [`chat`]

Page `/projects/<id>/chat`.
- **Test mode**: model dropdown → the `Q4_K_M` export → **LOAD** (≈ 10 s; header chip shows the model, API
  `/api/inference/status` shows `n_ctx 32768`, `layers 28/28`). Ask each ground-truth question. The page attaches the project's RAG
  corpus automatically ("5 source chunk(s) used as context"), so this is *grounded* answering; for pure recall use the API
  (`chat-v2` with `enabled_rag_ids: []`) — last run 7/12 pure recall vs 11/12 with retrieval.
- **Agent mode = the Guide** (`?mode=agent`, or the **? guide** button on *every* page): a docked panel in the page shell,
  so the conversation survives every page change (hard reloads restore it from `sessionStorage`). It uses the **configured
  helper model** (Gemma-12B), never silently switches to another loaded model; with no helper it shows the server's clear
  error. Tool calls stream in as they happen (`app_help`, `project_overview`, `recommend_training`, `dataset_health`,
  `system_status`, …) and each card shows a readable result with the raw JSON behind *Debug*. It can also act on the page,
  visibly: **navigate** to a page, **highlight** a control (pulse + "👉 label" tag), and **pre-fill** form fields (dashed cyan
  outline, editable, never submitted). Try, in order:
  1. "How do I train a model on my documents?" → `app_help` card cites the Training entry; it should open Training, pre-fill a
     preset and point at **Start training** — which you press yourself (no run may exist until you do).
  2. "What should I do next?" → the final reply must equal the readiness tool card's summary word for word.
  3. "Which settings should I use?" → `recommend_training` card: route, tier, epochs/rank/LR, step arithmetic, a **too few**
     warning when the dataset cannot reach the tier's step floor (then it should say "add data", not "train longer").
  4. "Where is pairs per chunk?" from another page → it opens Pairs first, then pulses `#prep-qpc`.
  5. "Create two Q&A pairs for aurora_spec_table.csv" → the only mutation (`create_qa_pairs`); pairs stay **pending**.
  It never approves, exports, trains, deletes or changes settings. A turn that exhausts its tool rounds still ends with an
  answer (forced final turn). Sandbox proof with a fake helper: `PYTHONPATH=src .venv/bin/python tests/e2e_guide_sandbox.py`
  (port 7896, temp cwd/HOME/FTS_ROOT/FTS_DB; screenshots in `.tmp/qa-shots/guide/`).

## 13. Clean up [`cleanup`]

Project overview → **DELETE** (confirm) → verify the project, its `output/projects/<id>`, `data/projects/<id>`, RAG corpus and
exported GGUFs are gone and the model dropdowns no longer offer them; delete the downloaded base model from the Model Library.
`nvidia-smi` back to baseline.

## What to look for on every page (cheap, always)

- No red console errors (`F12`), no failed network calls (Network tab, filter `4xx/5xx`).
- Every button gives feedback within ~1 s (spinner, toast, disabled state). A click that does nothing is a bug.
- Light **and** dark theme (🌙 toggle): text readable, no clipped table cells, nothing overlapping.
- Long jobs survive a page reload (progress is restored from the server, not lost).
- Numbers shown in the UI agree with the API and with `nvidia-smi`.

## Data-preparation routes — what the app supports vs industry standards

Pick the route from the **data shape and product need**, not from whichever algorithm sounds strongest: changing/citation-critical facts → RAG; curated prompt/answer examples → SFT; tool traces with schemas → tool SFT; pairwise judged responses → DPO; raw unlabelled domain text → continued pretraining; reviewed teacher traces → reasoning distillation. Routes can be combined (for example SFT + RAG); run a clean held-out evaluation before promotion.

| Route | Industry format (TRL "dataset formats") | App today |
|---|---|---|
| Instruction/QA SFT from documents | `messages` / ShareGPT `conversations` / Alpaca | **Yes** — export formats `sharegpt`, `alpaca`, `openai`; per-chunk generation, grounding filter, dedupe, coverage gate, optional retrieved-context rows |
| Prompt-completion SFT | `{"prompt":…, "completion":…}` | Via `alpaca`/`openai` export; no dedicated prompt-completion type |
| RAG-grounded answering | context in system/user turn | **Yes** — `grounded_share`, distractor chunks |
| Bring-your-own dataset | JSONL upload | **Yes** — Training page → *Upload my own* |
| Preference tuning (DPO) | `prompt` + `chosen` + `rejected` (standard strings or conversational messages) | **Yes** — Training tab → DPO; upload JSONL |
| ORPO / KTO / unpaired preferences | pairwise or labeled completion rows | **No** — this TRL build has no ORPO/KTO trainer; use DPO only for reviewed pairs |
| Tool-calling / agentic SFT | `messages` with `tool_calls` + `tools` JSON schema column | **Yes** — uploaded JSONL, standard SFT lifecycle, tool schemas passed to chat template |
| Continued pre-training (domain adaptation) | `{"text": …}` raw language modelling | **Yes** — raw text kept unwrapped; use held-out text/downstream eval, not the QA quiz |
| Reasoning / CoT distillation | teacher demonstrations in conversational `messages` | **Yes, as supervised distillation** — review traces; verify final task outcome independently |
| Quality: dedupe, decontamination, held-out split | MinHash near-dup, n-gram overlap vs eval, deterministic holdout | Exact/near-dup in export; deterministic seed-42 split; no eval decontamination |

References: [TRL dataset formats](https://huggingface.co/docs/trl/dataset_formats) and [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer); Gekhman et al. 2024, "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" (arXiv 2405.05904) — new facts are learned slowly and can raise hallucination, so keep a RAG path and an "I don't know" preference route for facts that must stay grounded.
