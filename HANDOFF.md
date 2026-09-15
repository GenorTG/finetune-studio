# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`): file library, data-prep chat, training/merge/GGUF export, benchmarks.
Edit on genorbox1 → push → fan-dragon pulls and runs it as the **systemd user unit `finetune-studio`** on :7860 (RTX 3090).

## State (verified 2026-09-15 10:06)
| Area | Status |
|------|--------|
| Git | `main` @ `8ff9fb0` (+ this docs commit). `626a48b` user-unit installer, `8ff9fb0` non-blocking upload report |
| fan-dragon service | ✅ `systemctl --user` `finetune-studio` active + enabled, linger=yes; :7860 pid `1713660` in `.../app.slice/finetune-studio.service` cgroup; legacy `finetune-studio-webui.service` removed; bare uvicorn `1701922` stopped by installer |
| Unit sandboxing | User unit has **no** ProtectHome/ProtectSystem (read-only home broke HF/triton caches). `--system` mode keeps hardening + needs sudo |
| Browser-tool upload | ✅ E2E on project `acd2e88f` (nightly-qa): `browser upload` of `/tmp/openclaw/uploads/velmaris_ember_college.md` via Upload trigger ref → file `dfeeee1ef56efcf7` in library table + data-prep source picker; inline `#fl-upload-status` = `Uploaded: 1 · Duplicates: 0 · Errors: 0` |
| Root cause of earlier upload "timeouts" | `flShowUploadReport` used native `alert()` → page blocked → tool call hung although the upload had landed. All 7 `alert()` in `data_prep.html` → `flReport()` (inline role=status + `fts.notify`) |
| Tests | ✅ 46 passed: `test_data_prep_upload_nonblocking` (3 new), `test_file_library_apis`, `test_install_diagnose`, `test_data_prep_promote`, `test_data_prep_route` |
| Ruff | New test file clean. `ruff check src/` = 502 errors, pre-existing (identical on stashed HEAD); no Python src touched |
| Probe data | nightly-qa library holds `velmaris_upload_probe.md` + `velmaris_ember_college.md` (test uploads, safe to trash) |
| Prior QA (QABUG-001..014) | Closed — details in `docs/archive/HANDOFF-2026-09-15.md` |

## Next steps
1. Replace remaining native dialogs that would block browser automation: `chat_v2.html:712,715`, `project_data.html:715`, `export_models.html:272,276`, `projects.html:295` fallback.
   `grep -rnP "(?<![\w.])alert\(|(?<![\w.])confirm\(" src/finetune_studio/webui/templates`
2. Same fix for other upload surfaces the browser may drive (`project_data.html` fb-upload, `project_training.html` dataset-upload, `projects.html` import) — verify each with `browser upload` + a file in `/tmp/openclaw/uploads/`.
3. Delete stale `scripts/finetune-studio.service` (system-unit template, unused by the installer) or regenerate it from `install-service.sh --system`.
4. Trash the two probe files from nightly-qa via the data-prep UI.
5. Chip away at the 502 pre-existing ruff errors per module (F821 first — real NameErrors).
   `.venv/bin/python -m ruff check src/ --select F821`

## Commands
- Tests: `.venv/bin/python -m pytest tests/test_data_prep_upload_nonblocking.py tests/test_file_library_apis.py -q`
- Lint: `.venv/bin/python -m ruff check src/`
- Deploy: `git push` then `ssh fan-dragon 'bash -lc "cd ~/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio"'`
- (Re)install unit: `ssh fan-dragon 'bash -lc "cd ~/finetune-studio && bash install-service.sh"'`
- Verify: `ssh fan-dragon 'bash -lc "systemctl --user is-active finetune-studio; ss -ltnp | grep 7860"'` + check pid cgroup (see `RESTART.md`)
- Logs: `ssh fan-dragon 'journalctl --user -u finetune-studio -n 50 --no-pager'`
- Browser upload: stage file in `/tmp/openclaw/uploads/`, `browser snapshot query=Upload` → `browser upload paths=[...] ref=<Upload ref>`

## Blockers
None.
