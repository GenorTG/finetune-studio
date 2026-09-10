# OCTOPUS-7741 — Internal Q4 Brief

> **Classification:** Internal — Engineering Eyes Only
> **Project Code:** OCTOPUS-7741
> **Issued:** 2026-09-10

## Summary

Project OCTOPUS-7741 covers the **Q4 reliability push** for the inference gateway. Specifically: keep model VRAM under 22 GiB steady-state on RTX 3090, ensure the unload path actually frees VRAM (regression fixed in commit `a34c08b`), and stabilize the data-prep chat tool-calling loop.

## Owner

- Primary: @genor
- Reviewer: @platform-infra
- Escalation: #octopus-7741

## Milestones

| Date | Milestone |
|------|-----------|
| 2026-09-08 | Unload-VRAM regression fix (a34c08b) |
| 2026-09-09 | External-API field hide fix (11d4cbc) |
| 2026-09-10 | E2E WebUI stress test (this doc's home project) |
| 2026-09-15 | Phase-3 Q&A pipeline parity review |

## Notes

The needle `OCTOPUS-7741` appears in every fixture for the 2026-09-10 E2E run so
the Q&A pipeline can be verified to surface project-specific facts that a base
model could not have memorized.
