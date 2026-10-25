"""Use-case templates: the four jobs where a LoRA adapter has measured wins over a prompted base model.

Evidence and plan: ``docs/audit/2026-10-24-training-research.md`` §0.3, §6 and §7.6. Each template says what data it
needs, turns the user's examples into SFT rows plus a held-out quiz the adapter never trains on, proposes training
settings that fit any safetensors chat model, and is measured the same way every time: the untrained base model and
the trained run answer the same held-out questions under the same instruction, and the per-question results are paired
(``testing/paired.py``, exact McNemar).

* ``templates`` — the four template specs (data needs, metric, settings).
* ``rows``      — validate the user's examples, shape SFT rows and held-out cases, seeded split.
* ``schema``    — the small JSON-schema subset an extraction template checks.
* ``metrics``   — exact match / field-F1 (extraction), accuracy / confusion matrix (classification).
* ``style_ab``  — blind A/B judge: which of two answers matches the house style (base vs trained, both orders).
* ``teacher``   — distillation: ask a big model (any provider row) for the answers the student learns.
* ``store``     — build a use case into a project (dataset + held-out suite + manifest).
* ``evaluate``  — base-vs-run evaluation through a compare group, scoring and the paired report.
"""
