# Finetune Studio — manual end-to-end test guide (user path)

How to test the app the way a person uses it: real browser, real files, real GPU. Every step says **which page**, **what to
click**, **what to wait for**, **what to check while it runs**, and **what counts as a pass**. The same path is automated in
`tests/e2e_user_walkthrough.py` (phase names in brackets) — run the script to reproduce, read this guide to understand or to
test by hand.

```bash
FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --list
FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --phase create --phase upload --phase prep
```

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

## 9–12. Test, benchmark, export, chat, agent  [`test`, `bench`, `gguf`, `chat`, `agent`]

(Documented here as they are verified by the script; see the phase docstrings.)

9. **Testing** `/projects/<id>/testing`: model *auto (latest merged)* + the auto-generated suite → **RUN**. Read the result
   split: plain rows are asked from memory; *grounded rows are asked with their own context* — never read the plain-row number as
   generalisation (it is recall of training facts). Held-out rows (seed-42 split) are the honest generalisation signal.
10. **Benchmarks** `/projects/<id>/benchmarks`: run an industry suite on base vs tuned and compare (regression check).
11. **Export** `/projects/<id>/export`: tick *gguf* + a quant. Prefer `q6_k`/`q4_k_m` for small models (see AGENTS.md Gotchas on
    Q8_0). Wait for the file; load it in **Chat**.
12. **Chat** `/projects/<id>/chat`: *test* mode — ask each ground-truth question without RAG (recall) and with RAG (grounded);
    *agent* mode — ask the model to operate the app (open a page, run a search) and check that the page switches and the UI
    reflects every action immediately.

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

| Route | Industry format (TRL "dataset formats") | App today |
|---|---|---|
| Instruction/QA SFT from documents | `messages` / ShareGPT `conversations` / Alpaca | **Yes** — export formats `sharegpt`, `alpaca`, `openai`; per-chunk generation, grounding filter, dedupe, coverage gate, optional retrieved-context rows |
| Prompt-completion SFT | `{"prompt":…, "completion":…}` | Via `alpaca`/`openai` export; no dedicated prompt-completion type |
| RAG-grounded answering | context in system/user turn | **Yes** — `grounded_share`, distractor chunks |
| Bring-your-own dataset | JSONL upload | **Yes** — Training page → *Upload my own* |
| Preference tuning (DPO/ORPO/KTO) | `prompt` + `chosen` + `rejected` (or unpaired + label) | **No** — SFT only |
| Tool-calling / agentic SFT | `messages` with `tool_calls` + `tools` JSON schema column | **No** |
| Continued pre-training (domain adaptation) | `{"text": …}` raw language modelling | **No** (raw text can only be wrapped as SFT) |
| Reasoning / CoT distillation | `messages` with `thinking`/analysis channel | **No** |
| Quality: dedupe, decontamination, held-out split | MinHash near-dup, n-gram overlap vs eval, 90/10 seed split | Exact/near-dup in export; seed-42 split; no eval decontamination |

References: TRL dataset formats (huggingface.co/docs/trl/dataset_formats); Gekhman et al. 2024, "Does Fine-Tuning LLMs on New
Knowledge Encourage Hallucinations?" (arXiv 2405.05904) — new facts are learned slowly and can raise hallucination, so keep a
RAG path and an "I don't know" preference route for facts that must stay grounded.
