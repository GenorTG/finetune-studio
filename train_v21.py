#!/usr/bin/env python3
"""
Chris AI v21 Training Script — knowledge preservation focus.

Changes from v20:
1. Data mixing: 97% v19 persona + 3% general knowledge (augmented)
2. Knowledge preservation: includes science, math, tech, refusal examples
3. Lower learning rate: 5e-5 (was 8e-5) to preserve more base knowledge
4. Same LoRA config: r=64, target modules same
"""
import os, sys, time, traceback
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "./data_v21_training.jsonl"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "./output_gemma4_v21"

print(f"[{time.strftime('%H:%M:%S')}] === Chris AI v21 Training ===")
print(f"[{time.strftime('%H:%M:%S')}] Dataset: {DATA_PATH}")
print(f"[{time.strftime('%H:%M:%S')}] Key changes from v20:")
print(f"  - Knowledge preservation: 97% persona + 3% general knowledge")
print(f"  - Lower LR: 5e-5 (was 8e-5) to preserve base knowledge")
print(f"  - Same LoRA config: r=64")
print("")

try:
    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    print(f"[{time.strftime('%H:%M:%S')}] Loading base model (Gemma 4 E4B 4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded")

    print(f"[{time.strftime('%H:%M:%S')}] Attaching LoRA adapters (r=64)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print(f"[{time.strftime('%H:%M:%S')}] LoRA attached")

    print(f"[{time.strftime('%H:%M:%S')}] Loading training data...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"[{time.strftime('%H:%M:%S')}] Dataset: {len(dataset)} examples")

    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = ""
            for i, msg in enumerate(messages):
                if i > 0:
                    text += "<turn|>\n"
                text += f"<turn>{msg['role']}\n{msg['content']}"
            text += "<turn|>\n<turn>model\n"
            texts.append(text)
        return {"text": texts}

    print(f"[{time.strftime('%H:%M:%S')}] Formatting conversations...")
    dataset = dataset.map(formatting_func, batched=True)

    print(f"[{time.strftime('%H:%M:%S')}] === STARTING TRAINING ===")
    print(f"[{time.strftime('%H:%M:%S')}] Epochs: 4 | Batch: 2 | Grad Accum: 4 | LR: 5e-5")
    print(f"[{time.strftime('%H:%M:%S')}] Cosine LR | Warmup 20 | WD 0.01")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=20,
            num_train_epochs=4,
            learning_rate=5e-5,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            save_strategy="no",
            report_to="none",
        ),
    )

    trainer.train()
    print(f"[{time.strftime('%H:%M:%S')}] === TRAINING COMPLETE ===")

    print(f"[{time.strftime('%H:%M:%S')}] Saving adapter...")
    model.save_pretrained(f"{OUTPUT_DIR}/final-adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final-adapter")
    print(f"[{time.strftime('%H:%M:%S')}] Adapter saved")

    print(f"[{time.strftime('%H:%M:%S')}] Exporting GGUF (Q4_K_M)...")
    model.save_pretrained_gguf(
        f"{OUTPUT_DIR}/export_gguf",
        tokenizer,
        quantization_method="q4_k_m",
    )
    print(f"[{time.strftime('%H:%M:%S')}] GGUF exported")
    print(f"[{time.strftime('%H:%M:%S')}] GGUF: {OUTPUT_DIR}/export_gguf/gemma-4-e4b-it.Q4_K_M.gguf")

except KeyboardInterrupt:
    print(f"\n[{time.strftime('%H:%M:%S')}] Interrupted")
except Exception as e:
    print(f"\n[{time.strftime('%H:%M:%S')}] ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
finally:
    print(f"\n[{time.strftime('%H:%M:%S')}] Done")
