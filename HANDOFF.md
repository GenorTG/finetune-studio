# Finetune Studio — Fresh Session Handoff (2026-09-10)

> **Supersedes:** any notes in OpenClaw session `agent:main:dashboard:1b010b16-…` ("Finetune Studio Work Cleanup").
> **Repo:** `~/work/finetune-studio/` on genorbox1 → `github.com/GenorTG/finetune-studio` → `/home/genortg/finetune-studio/` on fan-dragon (`finetune-studio.service`, port **7860**).

---

## Mission

Ship a reliable **local fine-tune + data-prep WebUI** with a full E2E browser test suite. Genor edits code **only on genorbox1**, commits/pushes to GitHub, then pulls on fan-dragon for GPU tests — **never SSH into fan-dragon to edit files**.

---

## Current state (verified 2026-09-10)

| Area | Status |
|------|--------|
| Git | `main` @ `84addd8`, clean, synced with origin |
| Unload VRAM | Fixed (`a34c08b`) — verify in browser after load |
| Data-prep tool-calling chat | Implemented (`data_prep_chat.py`, template updates) |
| 16k context + loader UI | Landed (`266649f`) |
| Workspace pollution | Removed from repo (`84addd8`) — **no project docs in `~/.openclaw/workspace/`** |
| Unit tests | **167 pass**, 37 fail (mostly `test_vram_profiler.py` — env/GPU dependent) |
| E2E WebUI suite | **Interrupted at Phase 2** — see Next steps |

---

## Genor's latest asks (prioritize)

1. **Dedicated Data Prep menu** with an **in-page tool-calling chat** (not the separate Inference tab). Chat loads a **local model** or **external OpenAI-compatible API** to help organize/extract training data from uploaded files.
2. **Unload button must actually free VRAM** — regression-test after every model load.
3. **Use OpenClaw planning tools** so Genor sees progress in the WebUI:
   - Call **`progress_card`** with a `plan` array at session start and update after each phase.
   - See [OpenClaw progress_card docs](https://docs.openclaw.ai/tools/progress-card).
4. **Complete E2E WebUI stress test** (browser tool, vision-check every step).

---

## E2E test plan (resume from Phase 2)

Fixture needle in every file: **`OCTOPUS-7741`**.

| Phase | Action |
|-------|--------|
| 1 | ✅ Cleanup / project delete UI (mostly done — re-verify badge = 0) |
| 2 | New project `2026-09-10-oftest`, vision-check |
| 3 | Data-prep: upload 10 fixture types (md/txt/xml/csv/pdf/doc/docx/jpg/png), load Qwen, run Q&A pipeline, approve/reject, export JSONL |
| 4 | Training tab: small base + JSONL, low epochs, **confirm Qwen unloads before train**, watch progress |
| 5 | Merge + export GGUF Q8_0, verify in model_exports + WebUI list |
| 6 | Benchmark trained vs base |
| 7 | Load exported GGUF, ask 3–5 questions only answerable via training data (OCTOPUS-7741) |
| 8 | Report bugs + summary |

**E2E rules:** browser tool only for test steps; vision-check after every action; fix inline; push/pull/restart fan-dragon as needed.

---

## Local docx fixture check (genorbox1)

Package root is **`src/`**, not repo root — `PYTHONPATH=.` fails with `ModuleNotFoundError`.

```bash
cd ~/work/finetune-studio
PYTHONPATH=src /tmp/fixtvenv/bin/python -c "
from finetune_studio.data.parsers.docx import parse
from pathlib import Path
r = parse(Path('tests/fixtures/octopus/spec.docx'))
print('needle', 'OCTOPUS-7741' in r['text'], 'parser', r['metadata'].get('parser'))
"
```

Dashboard may show exec commands as `PYTHON…tune-studio` — that is **UI truncation only** (`compactProgressText` / 120-char display). Never copy ellipsis into `exec` commands.

---

## Key paths

| What | Path |
|------|------|
| Repo root | `/home/genorbox1/work/finetune-studio/` |
| WebUI routes | `src/finetune_studio/webui/routes/` |
| Data prep chat | `src/finetune_studio/webui/routes/data_prep_chat.py` |
| Model manager | `src/finetune_studio/models/manager.py` |
| Update script | `update.sh` → `POST /api/system/update` |
| Tests | `tests/` (fixtures under `tests/fixtures/` if present) |
| Deploy target | fan-dragon: `finetune-studio.service` :7860 |

---

## Commands

```bash
# Local dev (genorbox1)
cd ~/work/finetune-studio
source .venv/bin/activate
pytest tests/ -q --ignore=tests/unit/test_vram_profiler.py
fts web --host 0.0.0.0 --port 7860

# Deploy flow
git add -A && git commit -m "..." && git push
# on fan-dragon: cd /home/genortg/finetune-studio && git pull && sudo systemctl restart finetune-studio
```

---

## Agent rules (non-negotiable)

- **File edits:** `read` → `edit` / `apply_patch` / `write`. **Never** `exec` Python/sed/awk to edit source files.
- **Docs live in this repo**, not `~/.openclaw/workspace/`.
- **Never restart OpenClaw gateway** without Genor's explicit approval.
- **Trash > rm** for cleanup.
- Use **`progress_card`** for multi-step work so the Control UI shows the plan.

---

## Known issues / watchlist

- Dashboard header may show `NO_DATA` in status chip (minor visual).
- External API fields must stay hidden when local model selected (`11d4cbc`).
- fan-dragon SSH flaky — if unreachable, run WebUI locally on genorbox1 for code-only work.

---

## Related docs

- `docs/INSTALL.md` — install/deploy
- `~/.openclaw/workspace/memory/projects/finetune-studio-full-pipeline.md` — pipeline phases (pointer only)
- `~/.openclaw/workspace/memory/projects/finetune-studio-qa.md` — prior QA notes
