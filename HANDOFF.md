# HANDOFF — finetune-studio

## Mission
Self-hosted local-LLM workshop for document prep, RAG, fine-tuning, evaluation, and chat.
Correctness is judged from actual data flow and reviewed transcripts, not optimistic UI or auto-scores.

## State (verified 2026-10-01)
| Area | State |
|---|---|
| Branch | `main` at `1c7790a`; origin/main matched at session start. Current work is uncommitted. |
| Inventory | 525 tracked repo files at audit start: 214 Python app modules, 180 test paths (28,385 lines), 15 scripts, 47 docs, plus UI/assets. Current untracked audit additions are separate. |
| Audit | App module lanes and UI/ops audits are documented; all 4,995 lines of `app.css` read. Test audit: 175/180 paths read as text/code (all 167 Python + 8 text/code paths); five PDF/image/Office fixtures were classified but not line-readable. |
| Fixes | RAG clear now removes text/indexes/metadata consistently; export detail/SSE/run-list enforce project ownership; UI injection/lifecycle/wizard defects, installer bounds, and QA-runner status/log handling fixed. |
| Test-quality | Install-diagnose JSON assertion parses output; optional-import smoke checks distinguish absent vs broken packages; live training/E2E drivers require explicit opt-in and report failed training; nightly runner is gated; MCP uninstall test uses isolated HOME/stub systemctl. |
| Public docs | README, install guide, tutorial, and Pages source corrected for Python 3.12–3.13, third-party/network caveats, and realistic dataset/model-learning claims. Not pushed. |
| Visibility | Repository is public; protected branches are public too. Existing module audit docs are already in the public tree. User decision on developer-doc publication/separation is pending. |
| Verification | Focused suites: 60, 44, and 13 passed in separate runs (overlapping; do not sum). Ruff on changed files passed. `bash -n`, `py_compile` for live scripts, and `git diff --check` passed. Full `ruff check src/` still reports 129 pre-existing findings (many public re-export facade F401s); no blind autofix. CODEMAP regeneration is deterministic; `make codemap-check` compares against HEAD and fails while the map is modified in this worktree. |

## Next steps
1. Finish validation after the latest E2E guard/test changes; classify the five non-text fixtures as the line-audit boundary.
2. Resolve developer-doc visibility/separation; do not push developer material to this public repo as private.
3. Inspect final docs/diff and HANDOFF; commit and push only content authorized for the selected visibility.

## Commands
- Focused route/RAG: `.venv/bin/python -m pytest tests/test_rag_audit.py tests/test_webui_routes_core_audit.py tests/test_webui_routes_workflow_audit.py -q`
- Lint: `.venv/bin/ruff check <changed-python-files>`
- E2E runner is external and sends Discord notifications; do not invoke during local checks.

## Blockers
- Awaiting user choice: keep developer docs unpublished until repository-private, or accept their visibility on a protected public dev branch.
- No remote `dev` branch existed at audit start; GitHub CLI unavailable.
