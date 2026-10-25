"""The four use-case templates: what each is for, what data it needs, how it is measured, and how to train it.

Example counts follow the literature in ``docs/audit/2026-10-24-training-research.md`` (§0.3, §6): behaviour tasks
show gains from 50-100 examples (OpenAI SFT guide) and typically use 50-1,000+ (LoRA Land: +34 points on average over
the prompted base; extraction studies +20-92 F1 over the same model without an adapter). Below the minimum a build is
refused; between the minimum and the recommended range it is built with a warning.

Training settings are model-agnostic: the app's measured LoRA defaults (all linear layers, rank 64 / alpha 128,
learning rate 1e-4: research §2 "LoRA Without Regret"), an effective batch of 8 (LoRA is less tolerant of large
batches), epochs chosen so a small set still gets enough optimizer steps, and a sequence length sized from the
longest example. Nothing depends on a model family.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TemplateId = Literal["extraction", "classification", "style", "distillation"]
Scoring = Literal["exact", "judge", "ab"]

HELDOUT_FRACTION = 0.2
MIN_HELDOUT = 20
MAX_HELDOUT = 200
MIN_PER_LABEL = 10           # classification: fewer examples of a label than this cannot train or test it
TARGET_STEPS = 60            # a small set trains for at least this many optimizer steps (within the epoch bounds)
MIN_EPOCHS, MAX_EPOCHS = 2, 5
BATCH_SIZE, GRAD_ACCUM = 2, 4  # effective batch 8
CHARS_PER_TOKEN = 3.5        # rough, any tokenizer; only sizes max_seq_length and the answer budget


@dataclass(frozen=True)
class DataNeeds:
    minimum: int                       # training examples (after the held-out split); fewer = refused
    recommended: tuple[int, int]       # where the literature measured gains
    evidence: str
    per_label_minimum: int = 0


@dataclass(frozen=True)
class Template:
    id: TemplateId
    title: str
    summary: str
    row_shape: dict[str, str]          # field -> meaning, for the upload help
    example_row: dict[str, Any]
    options: dict[str, str]            # optional build options -> meaning
    metric: str
    scoring: Scoring
    needs: DataNeeds
    default_instruction: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["needs"]["recommended"] = list(self.needs.recommended)
        out["heldout"] = {"fraction": HELDOUT_FRACTION, "minimum": MIN_HELDOUT, "maximum": MAX_HELDOUT}
        out["minimum_rows"] = self.needs.minimum + MIN_HELDOUT
        return out


_BEHAVIOUR_EVIDENCE = ("OpenAI SFT guide: gains show from 50-100 examples; LoRA Land (arXiv 2405.00732): +34 points "
                       "on average over the prompted base")

TEMPLATES: dict[str, Template] = {
    "extraction": Template(
        id="extraction",
        title="Strict output format / extraction",
        summary="Turn text into one JSON object with fixed fields (invoices, tickets, forms, records).",
        row_shape={"input": "the source text", "output": "the JSON object to extract (an object, or its JSON text)"},
        example_row={"input": "Invoice 2026-114 from Acme GmbH, due 12 March, total EUR 1,240.00",
                     "output": {"invoice_no": "2026-114", "vendor": "Acme GmbH", "total": 1240.0, "currency": "EUR"}},
        options={"schema": "optional JSON schema the outputs must follow (type, properties, required, enum, items, "
                           "additionalProperties)",
                 "instruction": "optional system instruction (default: reply with one JSON object only)"},
        metric="exact match of the whole object (pass) and field-F1 over every leaf value; also valid-JSON rate",
        scoring="exact",
        needs=DataNeeds(minimum=50, recommended=(200, 1000),
                        evidence=_BEHAVIOUR_EVIDENCE + "; extraction studies (PVminer arXiv 2603.05776, merchant "
                                 "extraction arXiv 2606.08051): +20-92 F1 over the same model without an adapter"),
        default_instruction="Extract the information from the user's text. Reply with one JSON object and nothing "
                            "else: no explanation, no code fence.",
    ),
    "classification": Template(
        id="classification",
        title="Classification / routing",
        summary="Put each text into one of a fixed set of labels (ticket routing, intent, triage, sentiment).",
        row_shape={"input": "the text to classify", "label": "the correct label"},
        example_row={"input": "My card was charged twice for one order.", "label": "billing"},
        options={"labels": "optional full label list (default: every label in the examples)",
                 "instruction": "optional system instruction (default: reply with exactly one label)"},
        metric="accuracy (pass = the right label) with a confusion matrix; also how often the reply is the label only",
        scoring="exact",
        needs=DataNeeds(minimum=50, recommended=(100, 1000), per_label_minimum=MIN_PER_LABEL,
                        evidence=_BEHAVIOUR_EVIDENCE + "; a fine-tuned 7B classified 78 % vs 59 % prompted (2025 case "
                                 "study, research §6). Aim for 50+ examples per label"),
        default_instruction="Classify the user's text. Reply with exactly one label from this list and nothing "
                            "else: {labels}",
    ),
    "style": Template(
        id="style",
        title="House style / tone",
        summary="Answer the way your team writes: tone, length, structure, wording habits.",
        row_shape={"prompt": "a request or question", "answer": "the answer written in the house style"},
        example_row={"prompt": "Can I change my delivery address?",
                     "answer": "Yes. Open Orders, pick the order and choose Change address. Done before dispatch."},
        options={"style_guide": "optional short description of the style; the base model is given it too, so the "
                                "comparison is against a prompted base",
                 "instruction": "optional system instruction (default: the style guide, if any)"},
        metric="blind A/B: an AI judge sees the base and the trained answer in both orders without knowing which is "
               "which and picks the one closer to the house-style reference; a win needs both orders to agree; sign "
               "test on wins vs losses",
        scoring="ab",
        needs=DataNeeds(minimum=50, recommended=(100, 500), evidence=_BEHAVIOUR_EVIDENCE + "; our Track B"),
        notes=["Style is not facts: an adapter trained on style does not learn your documents (use Smart Brain)."],
    ),
    "distillation": Template(
        id="distillation",
        title="Distil a big model for one task",
        summary="Teach a small, fast local model to answer one kind of request the way a big model does.",
        row_shape={"prompt": "a request of the task", "teacher_answer": "the big model's answer (or leave it out and "
                                                                          "pick a teacher provider to write it)"},
        example_row={"prompt": "Summarise this changelog for customers: ...",
                     "teacher_answer": "Three changes this week: ..."},
        options={"teacher": "name of the teacher model (recorded with the dataset)",
                 "teacher_provider_id": "provider row that writes missing teacher answers (Settings -> providers)",
                 "instruction": "optional system instruction used for teacher, base and student alike"},
        metric="agreement with the teacher: the AI judge checks each held-out answer against the teacher's answer "
               "(the teacher is the answer key); base and student are judged the same way",
        scoring="judge",
        needs=DataNeeds(minimum=100, recommended=(300, 1000),
                        evidence="Thinking Machines on-policy distillation (2025-10-27) and the OpenAI distillation "
                                 "guide; research §0.3: 50-1,000+ examples for one narrow task"),
        notes=["The student can only match the teacher on the task it was shown; check the held-out score."],
    ),
}


def get_template(template_id: str) -> Template:
    """The template, or ``KeyError`` naming the valid ids."""
    try:
        return TEMPLATES[template_id]
    except KeyError:
        raise KeyError(f"unknown use-case template {template_id!r}; one of: {', '.join(TEMPLATES)}") from None


def heldout_size(n_rows: int) -> int:
    """How many of ``n_rows`` examples are kept back as the held-out quiz."""
    return max(MIN_HELDOUT, min(MAX_HELDOUT, round(n_rows * HELDOUT_FRACTION)))


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text or "") / CHARS_PER_TOKEN)


def _pow2_at_least(n: int, lo: int, hi: int) -> int:
    size = lo
    while size < n and size < hi:
        size *= 2
    return min(size, hi)


@dataclass(frozen=True)
class TrainingPlan:
    """What to send to ``POST /api/training/start`` (with the dataset and the model), and how to evaluate."""

    preset_id: str
    training_mode: str
    system_prompt_mode: str
    overrides: dict[str, Any]
    eval_max_tokens: int
    eval_temperature: float
    reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def recommend_settings(template: Template, train_messages: list[list[dict[str, str]]]) -> TrainingPlan:
    """Settings for ``train_messages`` (each row's chat messages). Same rules for every model family."""
    n = len(train_messages)
    eff_batch = BATCH_SIZE * GRAD_ACCUM
    steps_per_epoch = max(1, math.ceil(n / eff_batch))
    epochs = min(MAX_EPOCHS, max(MIN_EPOCHS, math.ceil(TARGET_STEPS / steps_per_epoch)))
    longest = max((sum(estimate_tokens(m.get("content", "")) for m in msgs) for msgs in train_messages), default=0)
    seq_len = _pow2_at_least(int(longest * 1.25) + 32, 512, 4096)
    answers = [estimate_tokens(m.get("content", "")) for msgs in train_messages for m in msgs
               if m.get("role") == "assistant"]
    longest_answer = max(answers, default=0)
    if template.id == "classification":
        eval_tokens = 32
    else:
        floor = 128 if template.id == "extraction" else 256
        eval_tokens = min(2048, max(floor, int(longest_answer * 1.5) + 16))
    reasons = [
        (f"{n} training rows at an effective batch of {eff_batch} = {steps_per_epoch} steps per epoch; {epochs} "
         f"epochs (aim: about {TARGET_STEPS} steps, between {MIN_EPOCHS} and {MAX_EPOCHS} epochs)"),
        ("LoRA on every linear layer, rank 64 / alpha 128, learning rate 1e-4: the app's measured defaults "
         "(research §2: the best LoRA learning rate barely depends on rank)"),
        f"longest row about {longest} tokens -> max_seq_length {seq_len}",
        ("the instruction is part of every row (system turn), so system_prompt_mode is 'none': the base model, "
         "the trained run and the held-out quiz all see the same instruction"),
        "QLoRA (4-bit base) fits the most cards; pick a bf16 LoRA preset on the Training page when the model fits",
    ]
    return TrainingPlan(
        preset_id="qlora", training_mode="sft", system_prompt_mode="none",
        overrides={"num_epochs": epochs, "learning_rate": 1e-4, "lora_rank": 64, "lora_alpha": 128,
                   "batch_size": BATCH_SIZE, "gradient_accumulation_steps": GRAD_ACCUM, "max_seq_length": seq_len},
        eval_max_tokens=eval_tokens, eval_temperature=0.0, reasons=reasons,
    )
