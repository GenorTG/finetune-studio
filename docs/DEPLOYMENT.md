# Deployment & Update Pipeline

> How Finetune Studio gets installed, runs, and updates itself — on any host.
> The update pipeline is **self-healing**: a failed venv/deps state is repaired,
> not just reported. Two triggers: shell (`./update.sh`) or the web UI
> (**Settings → Updates** card, backed by `/api/system/update`).

## 1. First install

```bash
git clone https://github.com/GenorTG/finetune-studio && cd finetune-studio
./install.sh            # GPU-aware, idempotent
```

`install.sh` steps:
1. **detect_gpu** — `nvidia-smi` → driver→CUDA mapping (≥555 → cu132 prebuilt wheel,
   ≥525 → cu124, ≥520 → cu121, ≥470 → cu118, else CPU).
2. venv `.venv/` + `pip install -e .` (all core deps).
3. `llama-cpp-python` from the matching CUDA wheel index (never a source build
   unless no wheel exists).
4. llama.cpp CLI build into `LLAMA_CPP_DIR` (default project-local `.llama.cpp/`) —
   `llama-quantize` + `convert_hf_to_gguf.py` for the GGUF export endpoint.
5. systemd **user** service `finetune-studio.service` (port 7860, `--no-access-log`).
6. `scripts/install_diagnose.py` — deep health check (mixed installs, missing deps,
   broken torchaudio, service status); `--repair` autofixes.

## 2. Runtime layout

| Thing | Location |
|---|---|
| Repo + venv | `~/finetune-studio/` (or wherever you clone) |
| Service | `systemctl --user {status,restart} finetune-studio` |
| Logs | `journalctl --user -u finetune-studio -n 50` |
| Project data | `$FTS_ROOT` = `~/.finetune-studio/projects/<pid>/` |
| SQLite | `data/finetune_studio.db` (WAL) |
| Models | `models/gguf/`, HF cache `~/.cache/huggingface` |
| URL | `http://localhost:7860/` (or your host) |

## 3. The update pipeline

### 3.1 `update.sh` (repo root) — the worker

Six steps, each logged with `[HH:MM:SS]` prefixes (the API streams these lines):

1. `git pull --ff-only origin main` (warns + continues on divergence)
2. **venv health**: `scripts/install_diagnose.py` deep check → `--repair` autofix →
   last-resort venv recreate via `install.sh`
3. `pip install -e .` (dep sync)
4. llama.cpp CLI: build only if missing (`--no-llama` skips)
5. `init_db()` — idempotent schema migrations
6. `systemctl --user restart finetune-studio` (`--no-restart` skips)

Flags: `--check` (dry run: pull + migrations only, no pip/restart) ·
`--repair` (force venv recreate) · `--no-pull` · `--no-llama` · `--no-restart`.

> ⚠️ A restart **unloads the inference model** — reload via the Inference tab
> (or `POST /api/inference/load`) after updating.

### 3.2 HTTP API (`webui/routes/updates.py`)

| Endpoint | Purpose |
|---|---|
| `POST /api/system/update` | Queue a run. Body: `{mode: update|check|repair, no_pull, no_llama, no_restart, triggered_by}` → `{ok, update_id, status:"queued"}` |
| `GET /api/system/update/{uid}` | One attempt + `log_tail` (last 4 KB) |
| `GET /api/system/update/latest` | Current queued/running row — what the UI polls |
| `GET /api/system/updates?limit=50` | History |

Mechanics: FastAPI BackgroundTask spawns `update.sh` (`REPO_ROOT/update.sh`,
override `$FTS_UPDATE_SCRIPT`), streams stdout line-by-line into
`system_updates.log_text`. Statuses: `queued → running → done|error|cancelled`.
**Known quirk, handled:** a full update's step-6 restart kills its own streaming
worker — the row would be stuck `running` forever. On startup the service calls
`db.reconcile_stale_updates()`, which finalizes orphaned rows from the script's
own log (reached "Restarting finetune-studio.service" or "Update complete."
→ `done`; anything earlier → `error`). APPLY UPDATE therefore reports correctly
after the service comes back (~10 s).
Test mode: `FTS_SKIP_UPDATE=1` emits a canned log without spawning a shell —
used by smoke tests to assert the full lifecycle cheaply.

### 3.3 Web UI — Settings → Updates card

`Check` (dry run) · `Apply update` (full, confirm-gated, ~10 s downtime) ·
`Repair` (venv recreate, confirm-gated). Live log tail polls
`/api/system/update/latest` every 2 s, survives the restart (poll retries,
startup reconcile finalizes the row), shows the finished run's status + log
even when it completes between polls, and lists the last 5 attempts.
Code: `templates/settings.html` (card) + `static/js/settings.js` (logic).

## 4. Dev → deploy workflow (generic)

A common two-machine pattern:

```
edit + commit on the development machine
  → syntax check (py_compile / node --check)
  → git push origin main
  → on the GPU host: git pull --ff-only
  → systemctl --user restart finetune-studio   # if using the user unit
  → reload the model → verify live (API + browser)
```

Prefer editing only on the development machine and deploying via git pull on
the GPU host, so the running checkout stays a clean pull of `main`.

## 5. Health & troubleshooting

```bash
systemctl --user is-active finetune-studio
curl -s localhost:7860/api/inference/status        # {loaded, model, idle_seconds…}
curl -s localhost:7860/api/system/update/latest    # in-flight update, if any
scripts/install_diagnose.py --venv .venv           # full health report
```

| Symptom | Likely cause | Fix |
|---|---|---|
| `Failed to load model … not enough VRAM … top consumers: <process>(pid …)` | another process holds VRAM (load errors name consumers) | close/pause that process or use a smaller quant |
| Model loads but runs in RAM | VRAM too tight for `n_gpu_layers=99` | free VRAM; check `nvidia-smi` after load |
| UI shows stale css/js | asset version not bumped | bump `?v=N` in `base.html` |
| API 404 but route exists | catch-all registered first | move specific routes before `/{param}` |
| Update stuck `running` | worker died in the restart it triggered | fixed automatically: startup `reconcile_stale_updates()` finalizes it from the log |
| `git pull` fails only inside the service ("Bad owner or permissions on /etc/ssh/…") | the unit's sandbox (`ProtectSystem=full`, `ProtectHome=read-only`) breaks openssh's system-config ownership check | update.sh pulls with `GIT_SSH_COMMAND="ssh -F ~/.ssh/config"` (skips system config); repro: `systemd-run --user -p ProtectSystem=full -p ProtectHome=read-only` |

## 6. Rollback

```bash
git revert <sha> && git push          # then Settings → APPLY UPDATE
# or on the host:  ./update.sh --no-llama   (fast path: pull+deps+migrate+restart)
```

Related: `ARCHITECTURE.md` (what runs where) · `DEPENDENCIES.md` (dep policy) ·
`INSTALL.md` (first-install detail) · `REFACTOR-SPEC.md` (roadmap).
