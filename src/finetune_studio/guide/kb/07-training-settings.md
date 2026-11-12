---
id: training-settings
title: Choosing training settings (epochs, steps, rank, learning rate)
page: training
keywords: epochs steps optimizer rule of thumb how many epochs rank lora alpha learning rate batch size accumulation effective batch sequence length warmup preset recommend settings overfit underfit too few steps floor
---
## Purpose
Pick settings from the data and base model, not by habit. The preset advisor (`/api/training/recommend`) does the arithmetic and shows it.

## Workflow
1. **Steps, not epochs.** Domain recall is governed by real optimizer steps: steps = pairs × epochs ÷ effective batch, where effective batch = batch size × gradient accumulation. Evidence in the advisor's notes: a 515-pair factual dataset reached 95.1% strict recall at 772 steps (12 epochs at effective batch 8); about 257 steps reached 67%.
2. **Tier step floors:** Smoke none, Baseline (balanced) 400, Precision 700, Overkill 1000. The advisor raises epochs to clear the floor (cap 60) and warns when the data is too small to reach it — then add data rather than training longer, because longer just memorises.
3. **Dataset size scaling:** smaller datasets than the ~500-pair reference get more epochs (√(500 ÷ pairs), up to 4×); larger ones keep the tier's epochs (Baseline 6, Precision 12, Overkill 24), because every fact needs its passes however many facts there are. Measured on 4,306 facts: 4 passes per fact recalled 12.5 % of them, 12 passes 97 %; the old scaling gave Baseline 2 epochs (5.8 %) and Precision 4 (26.7 %). A big dataset therefore means a long run: steps grow with the data.
4. **Rank** scales with base size (≤1B halves it, ≤2B ×0.75, else unchanged); **learning rate** falls with size (≤7B keep 2e-4, ≤15B 1e-4, larger 8e-5).
5. Start with Baseline; if Testing (held-out) shows underfit, move to Precision. Overkill only when Precision fails on a verified dataset.
6. The advisor's step count ignores the validation split: the trainer keeps 10 % of the rows for validation, so the run's real total steps are about 10 % lower.

## Key controls
- `#training-preset` — pick the tier.
- `#preset-advisory` — shows step math and warnings.
- `[name="num_epochs"]` — epochs.
- `[name="lora_rank"]` — LoRA rank.
- `[name="learning_rate"]` — learning rate.
- `[name="batch_size"]` — batch size.
- `#gradient-accum-steps` — gradient accumulation.
- `#warmup-steps` — warmup steps (≈8% of the run is the advisor's choice).

## What to check
The advisory shows no "optimizer steps (floor …)" warning; loss falls steadily; held-out score on Testing, not training loss, decides quality.

## Common mistakes
Raising epochs on a tiny dataset to reach the floor (memorisation, not skill); a huge max sequence on a small GPU; using Overkill first.
