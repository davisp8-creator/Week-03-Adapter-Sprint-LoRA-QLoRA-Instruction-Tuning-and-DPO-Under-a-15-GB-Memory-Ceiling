"""
QLoRA fine-tune of a sub-1B causal LM (default: openai-community/gpt2) on the
synthetic 311 complaint -> dispatch-ticket JSONL produced by
generate_synthetic_complaints.py.

- 4-bit NF4 quantized base model (bitsandbytes) + LoRA adapters (peft).
- Loss is masked on prompt tokens: only the JSON ticket completion contributes
  to the loss, via labels=-100 on every prompt-token position.
- Peak GPU memory is printed (and logged to memory_log.jsonl) every
  `--log-every` steps, and training aborts loudly if peak memory crosses
  `--mem-limit-gb`.

Must run on a CUDA GPU (Colab T4). See env_setup.py to detect/install deps.

Usage:
    python scripts/train_qlora.py \
        --data data/311_complaints.jsonl \
        --model-name openai-community/gpt2 \
        --output-dir outputs/qlora-gpt2-311 \
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

try:
    from peft import prepare_model_for_kbit_training
except ImportError:  # older peft releases
    from peft import prepare_model_for_int8_training as prepare_model_for_kbit_training


PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Rewrite the resident complaint below into a structured 311 dispatch ticket. "
    "Respond with a single JSON object with exactly these fields: category, "
    "urgency (Low/Medium/High/Emergency), location, summary, and address "
    "(use null if no address is given).\n\n"
    "### Complaint:\n{complaint}\n\n"
    "### Ticket:\n"
)


def build_prompt(complaint: str) -> str:
    return PROMPT_TEMPLATE.format(complaint=complaint.strip())


class ComplaintTicketDataset(torch.utils.data.Dataset):
    """Tokenizes (prompt, completion) pairs with -100 labels over the prompt."""

    def __init__(self, path: str, tokenizer, max_length: int):
        self.examples: list[dict] = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                prompt = build_prompt(rec["complaint"])
                completion = json.dumps(rec["ticket"], ensure_ascii=False)

                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
                completion_ids = completion_ids + [tokenizer.eos_token_id]

                if len(prompt_ids) + len(completion_ids) > max_length:
                    skipped += 1
                    continue

                input_ids = prompt_ids + completion_ids
                labels = [-100] * len(prompt_ids) + completion_ids
                self.examples.append(
                    {
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": [1] * len(input_ids),
                    }
                )

        if skipped:
            print(f"Skipped {skipped} example(s) that exceed max_length={max_length} tokens")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def make_collate_fn(pad_token_id: int):
    def collate(batch: list[dict]) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, labels, attention_mask = [], [], []
        for ex in batch:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
            labels.append(ex["labels"] + [-100] * pad_len)
            attention_mask.append(ex["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    return collate


class GPUMemoryLimitExceeded(RuntimeError):
    pass


class MemoryGuardCallback(TrainerCallback):
    """Logs peak GPU memory on every Trainer log event and aborts over the ceiling."""

    def __init__(self, limit_gb: float, log_path: Path):
        self.limit_gb = limit_gb
        self.log_path = log_path
        self.start_time: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()
        self.log_path.write_text("", encoding="utf-8")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return control  # skip non-step logs (e.g. final train_runtime summary)

        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        elapsed = time.time() - self.start_time

        record = {
            "step": state.global_step,
            "loss": logs["loss"],
            "peak_mem_gb": round(peak_gb, 3),
            "elapsed_sec": round(elapsed, 1),
        }
        print(
            f"[step {state.global_step:>5}] loss={logs['loss']:.4f}  "
            f"peak GPU mem={peak_gb:.2f} GB  elapsed={elapsed:.1f}s"
        )
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if peak_gb > self.limit_gb:
            banner = "!" * 70
            print(banner)
            print(f"GPU MEMORY LIMIT EXCEEDED: {peak_gb:.2f} GB > {self.limit_gb:.2f} GB ceiling")
            print(banner)
            raise GPUMemoryLimitExceeded(
                f"Peak GPU memory {peak_gb:.2f} GB exceeded the {self.limit_gb:.2f} GB "
                f"ceiling at step {state.global_step}. Reduce --batch-size/--max-length "
                f"or raise --mem-limit-gb."
            )
        return control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/311_complaints.jsonl")
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument("--output-dir", default="outputs/qlora-gpt2-311")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--mem-limit-gb", type=float, default=14.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    # parse_known_args so this also runs unmodified inside Jupyter/Colab, which
    # injects its own "-f <kernel.json>" flag into sys.argv.
    args, _unknown = parser.parse_known_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA 4-bit training requires a CUDA GPU. Run this on the Colab T4 "
            "notebook (see env_setup.py) -- not the Windows CPU-only VM."
        )

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # GPT-2's attention block is a Conv1D (weight stored transposed vs nn.Linear),
    # hence fan_in_fan_out=True; c_attn is the fused q/k/v projection.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ComplaintTicketDataset(args.data, tokenizer, args.max_length)
    print(f"Loaded {len(train_dataset)} training examples from {args.data}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        logging_steps=args.log_every,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        optim="paged_adamw_8bit",
        report_to=[],
        seed=args.seed,
    )

    memory_log_path = output_dir / "memory_log.jsonl"
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
        callbacks=[MemoryGuardCallback(args.mem_limit_gb, memory_log_path)],
    )

    start = time.time()
    try:
        trainer.train()
    except GPUMemoryLimitExceeded:
        print("Training aborted: GPU memory ceiling breached. See memory_log.jsonl for the trace.")
        raise
    total_elapsed = time.time() - start

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Training complete in {total_elapsed / 60:.1f} min. Peak GPU memory: {peak_gb:.2f} GB")

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    summary = {
        "model_name": args.model_name,
        "dataset": args.data,
        "num_examples": len(train_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "peak_gpu_mem_gb": round(peak_gb, 3),
        "mem_limit_gb": args.mem_limit_gb,
        "wall_clock_sec": round(total_elapsed, 1),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
