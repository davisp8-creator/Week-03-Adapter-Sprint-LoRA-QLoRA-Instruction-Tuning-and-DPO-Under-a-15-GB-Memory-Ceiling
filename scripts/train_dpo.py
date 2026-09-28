"""
DPO pass (trl) on top of the QLoRA adapter produced by train_qlora.py, using
the frozen SFT policy as the reference model.

The base model is loaded once, 4-bit NF4 quantized, and the saved SFT adapter
is attached as the trainable ("default") LoRA adapter. `ref_model` is left as
None: since the model is a PEFT model with a *pretrained* adapter, trl clones
it into a second, frozen "ref" adapter sharing the same quantized base
weights -- this is the frozen reference policy, with no second full model
copy loaded into memory.

- `--beta` controls the DPO temperature (KL penalty strength vs the frozen
  reference).
- The implicit reward margin (rewards/margins = beta * (chosen_logratio -
  rejected_logratio)), plus rewards/chosen, rewards/rejected and
  rewards/accuracies, are printed and logged every `--log-every` steps
  alongside peak GPU memory and wall-clock elapsed time.
- Training aborts loudly if peak GPU memory crosses `--mem-limit-gb`.

Must run on a CUDA GPU (Colab T4). See env_setup.py to detect/install deps.

Usage:
    python scripts/train_dpo.py \
        --sft-adapter outputs/qlora-gpt2-311/adapter \
        --data data/dpo_address_preferences.jsonl \
        --output-dir outputs/dpo-gpt2-311 \
        --beta 0.1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import transformers
import trl
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import DPOConfig, DPOTrainer

try:
    from peft import prepare_model_for_kbit_training
except ImportError:  # older peft releases
    from peft import prepare_model_for_int8_training as prepare_model_for_kbit_training

from compat import build_config
from instrumentation import GPUMemoryLimitExceeded, InstrumentationCallback

REWARD_METRIC_KEYS = ("rewards/margins", "rewards/chosen", "rewards/rejected", "rewards/accuracies")


def load_preference_dataset(path: str) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-adapter", default="outputs/qlora-gpt2-311/adapter",
                         help="path to the LoRA adapter saved by train_qlora.py")
    parser.add_argument("--data", default="data/dpo_address_preferences.jsonl")
    parser.add_argument("--model-name", default="openai-community/gpt2",
                         help="must match the base model the SFT adapter was trained on")
    parser.add_argument("--output-dir", default="outputs/dpo-gpt2-311")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature (KL penalty strength)")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--mem-limit-gb", type=float, default=14.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    # parse_known_args so this also runs unmodified inside Jupyter/Colab, which
    # injects its own "-f <kernel.json>" flag into sys.argv.
    args, _unknown = parser.parse_known_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DPO training requires a CUDA GPU. Run this on the Colab T4 "
            "notebook (see env_setup.py) -- not the Windows CPU-only VM."
        )

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"transformers=={transformers.__version__}  trl=={trl.__version__}  torch=={torch.__version__}")

    print(f"Loading tokenizer from SFT adapter: {args.sft_adapter}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    print(f"Loading base model: {args.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model = prepare_model_for_kbit_training(base_model)

    print(f"Attaching pretrained SFT adapter: {args.sft_adapter}")
    model = PeftModel.from_pretrained(base_model, args.sft_adapter, is_trainable=True)
    model.print_trainable_parameters()

    train_dataset = load_preference_dataset(args.data)
    print(f"Loaded {len(train_dataset)} preference pairs from {args.data}")

    steps_per_epoch = -(-len(train_dataset) // (args.batch_size * args.grad_accum))  # ceil div
    total_train_steps = max(1, round(steps_per_epoch * args.epochs))

    dpo_config = build_config(
        DPOConfig,
        total_train_steps=total_train_steps,
        beta=args.beta,
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        max_length=args.max_length,
        gradient_checkpointing=True,
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
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT model + pretrained adapter -> trl clones a frozen "ref" adapter
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[InstrumentationCallback(args.mem_limit_gb, memory_log_path, extra_keys=REWARD_METRIC_KEYS)],
    )

    start = time.time()
    try:
        trainer.train()
    except GPUMemoryLimitExceeded:
        print("Training aborted: GPU memory ceiling breached. See memory_log.jsonl for the trace.")
        raise
    total_elapsed = time.time() - start

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"DPO training complete in {total_elapsed / 60:.1f} min. Peak GPU memory: {peak_gb:.2f} GB")

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    summary = {
        "model_name": args.model_name,
        "sft_adapter": args.sft_adapter,
        "dataset": args.data,
        "num_examples": len(train_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "beta": args.beta,
        "peak_gpu_mem_gb": round(peak_gb, 3),
        "mem_limit_gb": args.mem_limit_gb,
        "wall_clock_sec": round(total_elapsed, 1),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
