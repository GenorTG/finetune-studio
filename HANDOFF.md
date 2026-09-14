# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as a bare `uvicorn` on port 7860 (RTX 3090). No systemd by default.

## State (verified 2026-09-14 15:15)
| Area | Status |
|------|--------|
| Git | `main` — data-prep preview tabs (RAW/PARSED/VERSIONS/CONVERSIONS) + parsed-MD column icon |
| Data-prep file library preview | ✅ Tabbed modal in `data_prep.html`: RAW (existing image/pdf/text), PARSED (sibling-.md heuristic + banner; **no** `/files/{fid}/parsed` API), VERSIONS (`/{fid}/versions`), CONVERSIONS timeline (`/{fid}/conversions`). Version pill shows 📝 / · for parsed readiness. |
| Parsed-MD API gap | ⚠️ No `GET /files/{fid}/parsed` — converted bytes live at `file_conversions.converted_path` / legacy `files/<sha>/parsed.txt`. UI documents TODO + banner; did not invent an endpoint. |
| UI audit Phase 2 item 1 | ✅ Data-prep parsed-preview + versions + conversions |
| Pytest (genorbox1) | ✅ `test_data_editor` + `test_project_data_browser` + `test_project_dashboard` = 12 passed |
| E2E suite | ⚠️ Stopped at Phase 3 (project `b08426e3`); Phases 3–8 pending |
| Known viewport gap | ⚠️ 1920×1080 screenshot unreachable this session |
| Backfill old training-run errors | ⚠️ `bdc217b1` + 2 others still show empty `error` columns |

## Next steps
1. **fan-dragon verify** — `git pull --ff-only`; restart per `RESTART.md`; open `http://fan-dragon:7860/projects/b08426e3/data-prep`; click 👁 → confirm 4 tabs.
2. **UI audit Phase 2 remaining** — training promote-to-production button (`POST /api/projects/{pid}/promote`); testing suite dropdown on `/projects/{pid}/testing`.
3. Optional: add `GET /api/projects/{pid}/files/{fid}/parsed` (or `/converted?format=md`) so PARSED tab can stream real converted MD.
4. Resume E2E Phase 3 — `tests/run_qa.sh`; upload 10 `OCTOPUS-7741` fixtures to project `b08426e3`.
5. Backfill `training_runs.error` for the 3 pre-`daf9fa1` failed runs from `/tmp/uvicorn.log`.
6. Wire optional APIs: `PATCH /api/projects/{pid}/files/{fid}/rename` + per-file hard-purge.
7. Optionally `bash scripts/install-service.sh` on fan-dragon for systemd on :7860.

## Commands
- Test (genorbox1): `.venv/bin/python -m pytest tests/test_data_editor.py tests/test_project_data_browser.py tests/test_project_dashboard.py -q -p no:cacheprovider`
- Ruff (touched routes): `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/file_library.py --select F821,F401,E,W`
- Deploy: `git push`, then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart per `RESTART.md` (start new → kill old pid → `ss -ltnp | grep 7860`)

## Blockers
- genorbox1 is dev-only: pure unit tests + ruff here; visual/E2E/GPU only on fan-dragon after pull + restart
- Parsed MD content cannot be served through existing file-library routes (only `/raw`) — PARSED tab uses heuristics + explicit banner
