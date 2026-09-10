# E2E Test Plan — finetune-studio · 2026-09-10

Mirror of the live `progress_card` plan, also committed for persistent visibility
in case the OpenClaw progress card is stale or the gateway is rate-limiting writes.

## Status

| Step | State | Notes |
|------|-------|-------|
| Read HANDOFF + verify git status | ✅ done | `main` @ `84addd8`, clean |
| Debug progress_card | ✅ done | intermittent gateway validation; retryable |
| Phase 2: verify project 2026-09-10-oftest | ✅ done | pre-existing from prior session; id `b08426e3` |
| **Phase 3: create + upload 10 fixture files** | **▶ in_progress** | md/txt/xml/csv/pdf/doc/docx/jpg/png/html |
| Phase 3 cont.: run Qwen Q&A pipeline, export JSONL | ⏳ pending | Qwen3.8-27B-abliterated-Q4_K_M already loaded at 78.8% VRAM |
| Phase 4: training (small base + JSONL, low epochs, confirm Qwen unloads) | ⏳ pending | RTX 3090 24GB |
| Phase 5: merge LoRA + export GGUF Q8_0 | ⏳ pending | verify in `model_exports` + WebUI list |
| Phase 6: benchmark trained vs base | ⏳ pending | |
| Phase 7: load exported GGUF; 3-5 questions from OCTOPUS-7741 | ⏳ pending | only answerable via training data |
| Phase 8: bug report + summary, commit/push | ⏳ pending | |
| Re-verify Phase 1 cleanup: "projects [2]" nav badge | ⏳ pending | deferred, not blocking |

## Fixture needle

Every fixture file (10 total) embeds the string **`OCTOPUS-7741`** so we can
prove the Q&A pipeline only generates answers a base model could not produce.

## progress_card flakiness note

`openclaw:core:progress_card` has been intermittently rejecting valid payloads
during this session with `plan: must be array` (despite plan being a real
array). A smoke test (`{"plan":[{"status":"in_progress","step":"test"}]}`)
worked once, then identical follow-ups failed. This is a gateway-side
validation flakiness issue, not a payload shape issue — when the same shape
succeeds, the full plan updates correctly. This file is the durable backup.
